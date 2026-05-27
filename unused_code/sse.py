import asyncio
from tokenizer.tokenizer_service import tokenizer_service

# O(n2) decoding for now, can be optimized with sliding window or optimized tokenizer buffer
async def stream_tokens(generator):

    queue = asyncio.Queue()

    def producer():
        for token in generator:
            asyncio.run_coroutine_threadsafe(queue.put(token), loop)
        asyncio.run_coroutine_threadsafe(queue.put(None), loop)

    loop = asyncio.get_event_loop()
    asyncio.get_event_loop().run_in_executor(None, producer)

    tokens = []
    prev_text = ""

    while True:
        token = await queue.get()

        if token is None:
            break

        tokens.append(token)

        current_text = tokenizer_service.decode(tokens)
        delta = current_text[len(prev_text):]
        prev_text = current_text

        if delta:
            yield {"data": delta}

    yield {"data": "[DONE]"}
