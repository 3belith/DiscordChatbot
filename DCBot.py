from __future__ import annotations

import asyncio
import errno
import logging
import os
import random
import sys
import time
from datetime import timedelta
from pathlib import Path

import discord
from dotenv import load_dotenv

from ai import LilpaAI
from memory import ConversationMemory


# ============================================================
# 기본 설정
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

load_dotenv(BASE_DIR / ".env")

DISCORD_TOKEN = os.getenv("DISCORD_TOKEN")

if not DISCORD_TOKEN:
    raise RuntimeError("DISCORD_TOKEN이 없습니다.")


# ============================================================
# Discord
# ============================================================

intents = discord.Intents.default()
intents.message_content = True

bot = discord.Client(intents=intents)


# ============================================================
# AI / 메모리
# ============================================================

ai = LilpaAI()

memory = ConversationMemory(
    maxlen=10,
    recent_turns=4,
)


# ============================================================
# 상태
# ============================================================

cooldowns: dict[int, float] = {}

processing_messages: set[int] = set()

summary_tasks: set[asyncio.Task[None]] = set()

summary_locks: dict[int, asyncio.Lock] = {}


# ============================================================
# 설정값
# ============================================================

MAX_GEMINI_CONCURRENCY = int(
    os.getenv("GEMINI_MAX_CONCURRENCY", "4")
)

gemini_semaphore = asyncio.Semaphore(
    MAX_GEMINI_CONCURRENCY
)

COOLDOWN_SECONDS = 1.0

# Discord 메시지 최대 길이
MAX_CHARS = 2000

# Gemini에 넣는 최근 대화 최대 단어 수
MAX_PROMPT_WORDS = 180

# 검열 시 타임아웃
MOD_TIMEOUT_SECONDS = int(
    os.getenv("MOD_TIMEOUT_SECONDS", "30")
)


# ============================================================
# 로그
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

logger = logging.getLogger(__name__)


# ============================================================
# 빈 멘션 응답
# ============================================================

EMPTY_MESSAGES = [
    "왜 불렀어?",
    "할 말 있어?",
    "듣고 있어.",
    "말해 봐.",
    "부른 거 아니었어?",
]


# ============================================================
# 쿨다운
# ============================================================

def is_cooldown(user_id: int) -> bool:
    now = time.time()

    last = cooldowns.get(user_id, 0.0)

    if now - last < COOLDOWN_SECONDS:
        return True

    cooldowns[user_id] = now

    return False


# ============================================================
# 멘션 제거
# ============================================================

def get_question(message: discord.Message) -> str:
    if bot.user is None:
        return message.content.strip()

    return (
        message.content
        .replace(f"<@{bot.user.id}>", "")
        .replace(f"<@!{bot.user.id}>", "")
        .strip()
    )


# ============================================================
# Gemini 프롬프트
# ============================================================

def make_prompt(
    channel_id: int,
    username: str,
    question: str,
) -> str:

    context = memory.build_context(
        channel_id,
        max_words=MAX_PROMPT_WORDS,
    )

    return (
        "[현재 대화 상대]\n"
        f"이름: {username}\n\n"

        "[최근 대화 맥락]\n"
        f"{context}\n\n"

        "[현재 메시지]\n"
        f"{username}: {question}"
    )


# ============================================================
# 대화 요약
# ============================================================

async def update_summary(channel_id: int) -> None:

    lock = summary_locks.setdefault(
        channel_id,
        asyncio.Lock(),
    )

    async with lock:

        items = memory.get_summary_request(
            channel_id
        )

        if not items:
            return

        try:

            async with gemini_semaphore:

                summary = await asyncio.to_thread(
                    ai.generate_summary,
                    items,
                )

        except Exception:

            logger.exception(
                "Conversation summary failed for channel %s",
                channel_id,
            )

            summary = None

        memory.complete_summary(
            channel_id,
            summary,
            items,
        )


def schedule_summary(channel_id: int) -> None:

    task = asyncio.create_task(
        update_summary(channel_id)
    )

    summary_tasks.add(task)

    task.add_done_callback(
        summary_tasks.discard
    )


# ============================================================
# 긴 메시지 분할
# ============================================================

def chunk_text(
    text: str,
    size: int = MAX_CHARS,
) -> list[str]:

    return [
        text[index:index + size]
        for index in range(
            0,
            len(text),
            size,
        )
    ]


# ============================================================
# 정상 답변 전송
# ============================================================

async def send_answer(
    message: discord.Message,
    text: str,
) -> None:

    prefix = f"{message.author.mention}\n"

    first_limit = MAX_CHARS - len(prefix)

    # 한 번에 보낼 수 있는 경우
    if len(text) <= first_limit:

        await message.reply(
            f"{prefix}{text}",
            mention_author=False,
        )

        return

    # 첫 번째 메시지
    first_part = text[:first_limit]

    await message.reply(
        f"{prefix}{first_part}",
        mention_author=False,
    )

    # 나머지 메시지
    remaining = text[first_limit:]

    for part in chunk_text(
        remaining,
        MAX_CHARS,
    ):

        await message.channel.send(part)


# ============================================================
# 검열 처리
# ============================================================

async def handle_moderation(
    message: discord.Message,
    answer: str,
) -> None:

    # <MOD> 제거
    warning = answer[len("<MOD>"):].strip()

    # AI가 멘트를 비워서 보내는 경우
    if not warning:

        warning = (
            "그런 표현은 쓰지 마. "
            "조금 예쁘게 말하자."
        )

    # --------------------------------------------------------
    # 원본 메시지 삭제
    # --------------------------------------------------------

    try:

        await message.delete()

    except discord.NotFound:
        # 이미 삭제된 경우
        pass

    except discord.Forbidden:

        logger.warning(
            "메시지 삭제 권한이 없습니다."
        )

    except discord.HTTPException:

        logger.exception(
            "검열 메시지 삭제 실패"
        )

    # --------------------------------------------------------
    # 타임아웃
    # --------------------------------------------------------

    try:

        # 봇 자신 / 서버 관리자 등은 Discord 권한 구조상
        # 타임아웃이 실패할 수 있음
        if isinstance(
            message.author,
            discord.Member,
        ):

            await message.author.timeout(
                discord.utils.utcnow()
                + timedelta(
                    seconds=MOD_TIMEOUT_SECONDS
                ),
                reason="AI 검열",
            )

    except discord.Forbidden:

        logger.warning(
            "사용자 타임아웃 권한이 없습니다."
        )

    except discord.NotFound:

        logger.warning(
            "타임아웃 대상 사용자를 찾을 수 없습니다."
        )

    except discord.HTTPException:

        logger.exception(
            "사용자 타임아웃 실패"
        )

    # --------------------------------------------------------
    # 경고 메시지
    # --------------------------------------------------------

    await message.channel.send(
        f"{message.author.mention} {warning}",
        allowed_mentions=discord.AllowedMentions(
            users=True
        ),
    )


# ============================================================
# Discord 준비 완료
# ============================================================

@bot.event
async def on_ready() -> None:

    print(
        f"{bot.user} 실행 완료"
    )


# ============================================================
# Discord 오류
# ============================================================

@bot.event
async def on_error(
    event: str,
    *args: object,
    **kwargs: object,
) -> None:

    error = sys.exc_info()[1]

    # 파일이 없는 경우 Discord 이벤트 전체가
    # 시끄럽게 로그에 찍히는 것을 방지
    if (
        isinstance(error, OSError)
        and error.errno == errno.ENOENT
    ):
        return

    logger.exception(
        "Discord event failed: %s",
        event,
    )


# ============================================================
# 메시지 처리
# ============================================================

@bot.event
async def on_message(
    message: discord.Message,
) -> None:

    # --------------------------------------------------------
    # 봇 메시지 무시
    # --------------------------------------------------------

    if message.author.bot:
        return

    # --------------------------------------------------------
    # 봇 멘션이 없으면 무시
    # --------------------------------------------------------

    if bot.user not in message.mentions:
        return

    # --------------------------------------------------------
    # 중복 처리 방지
    # --------------------------------------------------------

    if message.id in processing_messages:
        return

    processing_messages.add(
        message.id
    )

    # --------------------------------------------------------
    # 쿨다운
    # --------------------------------------------------------

    if is_cooldown(
        message.author.id
    ):

        try:

            await message.reply(
                "ㄱㄷ",
                mention_author=False,
            )

        finally:

            processing_messages.discard(
                message.id
            )

        return

    # --------------------------------------------------------
    # 질문 추출
    # --------------------------------------------------------

    question = get_question(
        message
    )

    # --------------------------------------------------------
    # 멘션만 한 경우
    # --------------------------------------------------------

    if not question:

        try:

            await message.reply(
                f"{message.author.mention} "
                f"{random.choice(EMPTY_MESSAGES)}",
                mention_author=False,
            )

        finally:

            processing_messages.discard(
                message.id
            )

        return

    # ========================================================
    # 실제 AI 처리
    # ========================================================

    try:

        username = message.author.display_name

        channel_id = message.channel.id

        # ----------------------------------------------------
        # 프롬프트 생성
        # ----------------------------------------------------

        prompt = make_prompt(
            channel_id,
            username,
            question,
        )

        # ----------------------------------------------------
        # Gemini 호출
        # ----------------------------------------------------

        async with message.channel.typing():

            async with gemini_semaphore:

                answer = await asyncio.to_thread(
                    ai.generate,
                    prompt,
                )

        # ====================================================
        # ★ AI 검열
        # ====================================================

        if answer.lstrip().startswith("<MOD>"):

            # 앞에 공백이 있어도 정상 처리
            answer = answer.lstrip()

            await handle_moderation(
                message,
                answer,
            )

            return

        # ====================================================
        # 정상 답변
        # ====================================================

        memory.append(
            channel_id,
            username,
            question,
            answer,
        )

        await send_answer(
            message,
            answer,
        )

        # ----------------------------------------------------
        # 주기적인 대화 요약
        # ----------------------------------------------------

        schedule_summary(
            channel_id
        )

    # ========================================================
    # 오류 처리
    # ========================================================

    except Exception as exc:

        logger.exception(
            "Message processing failed"
        )

        try:

            await message.reply(
                f"오류: {exc}",
                mention_author=False,
            )

        except Exception:

            logger.exception(
                "오류 메시지 전송 실패"
            )

    finally:

        processing_messages.discard(
            message.id
        )


# ============================================================
# 실행
# ============================================================

def run_bot() -> None:

    bot.run(
        DISCORD_TOKEN
    )


if __name__ == "__main__":

    run_bot()