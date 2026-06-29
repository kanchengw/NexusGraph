import asyncio
import selectors
import os

os.environ.setdefault("APP_ENV", "development")

asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
loop = asyncio.SelectorEventLoop()
asyncio.set_event_loop(loop)

import uvicorn
uvicorn.run("app.main:app", host="0.0.0.0", port=8000)
