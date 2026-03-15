import asyncio
from httpx import AsyncClient

async def main():
    async with AsyncClient() as client:
        resp = await client.get('https://example.com')
        print(resp.status_code)

if __name__ == '__main__':
    asyncio.run(main())
