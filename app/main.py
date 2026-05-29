
import asyncio

from telegram import Update
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    ContextTypes,
    filters
)

from top_chunks.retriever import TopChunksRetriever
from llm.giga_generator import generate_rag_answer

from utils.logger import (
    log_question,
    log_chunks
)


TOKEN = "8101923743:AAEMx1AzEu6kvzqzg6zJ62D6TZIjQoHBVRM"


# ---------------- RAG PIPELINE ----------------

def run_rag(question: str) -> str:

    # save question
    log_question(question)

    retriever = TopChunksRetriever()

    # retrieve chunks
    chunks = retriever.get_top_chunks(question)

    # save retrieved chunks
    

    # sort by score
    chunks = sorted(
        chunks,
        key=lambda x: x.get("score", 0),
        reverse=True
    )[:5]
    log_chunks(question, chunks)

    # generate answer
    return generate_rag_answer(
        query=question,
        chunks=chunks
    )


# ---------------- HANDLERS ----------------

async def start(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE
):

    await update.message.reply_text(
        "Привет 👋 Я RAG-бот по поступлению. Задай вопрос."
    )


async def handle_message(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE
):

    user_text = update.message.text

    await update.message.reply_text(
        "⏳ Думаю..."
    )

    try:
        answer = run_rag(user_text)

    except Exception as e:
        answer = f"Ошибка: {str(e)}"

    await update.message.reply_text(answer)


# ---------------- MAIN ----------------

def main():

    app = Application.builder().token(TOKEN).build()

    app.add_handler(
        CommandHandler("start", start)
    )

    app.add_handler(
        MessageHandler(
            filters.TEXT & ~filters.COMMAND,
            handle_message
        )
    )

    print("Bot started...")

    app.run_polling()


if __name__ == "__main__":
    main()
