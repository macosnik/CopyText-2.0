from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, ContextTypes
import os, json

with open("token.json") as f:
    TOKEN = json.load(f).get("token")

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("/dataset - получить датасет")

async def send_dataset(update: Update, context: ContextTypes.DEFAULT_TYPE):
    file_path = "dataset.csv"
    if os.path.exists(file_path):
        await update.message.reply_document(document=open(file_path, "rb"))
    else:
        await update.message.reply_text("dataset.csv не найден")

def main():
    app = ApplicationBuilder().token(TOKEN).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("dataset", send_dataset))

    app.run_polling()

if __name__ == "__main__":
    main()
