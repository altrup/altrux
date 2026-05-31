import asyncio
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.model.loader import registry

async def main():
    await registry.startup()
    registry.reset_session("Hello")

    for i in range(3):
        t = time.time()
        result = registry.generate_one_token()
        elapsed = time.time() - t
        print(f"call {i}: {elapsed:.2f}s  token={repr(result['generated_token'])}")


asyncio.run(main())
