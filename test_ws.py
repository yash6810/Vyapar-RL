import asyncio
from websockets.asyncio.client import connect

async def main():
    try:
        async with connect("ws://localhost:9999") as ws:
            pass
    except BaseException as e:
        print(f"Caught: {type(e)}")

asyncio.run(main())
