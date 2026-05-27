import asyncio
from scheduler.request_queue import request_queue


class BatchScheduler:
    def __init__(self, engine, tokenizer):
        self.engine = engine
        self.tokenizer =  tokenizer

        self.batch_size = 4
        self.timeout = 0.01

    async def run(self):
        while True:
            batch = []

            # 1. wait for request
            req = await request_queue.get()
            batch.append(req)

            # 2. collect more requests
            start_time = asyncio.get_event_loop().time()

            while len(batch) < self.batch_size:
                if (asyncio.get_event_loop().time() - start_time) > self.timeout:
                    break
                try:
                    req = await asyncio.wait_for(
                        request_queue.get(),
                        timeout=self.timeout
                    )
                    batch.append(req)
                except asyncio.TimeoutError:
                    break
                    
            # 3. process batch
            try:
                await self.process_batch(batch)
            except Exception as e:
                for req in batch:
                    if not req.future.done():
                        req.future.set_exception(e)

    async def process_batch(self, batch):
        prompts = [r.prompt for r in batch]
        
        encoded = self.tokenizer.tokenizer(
            prompts,
            return_tensors='pt',
            padding=True,
            truncation=True
        )

        input_ids = encoded["input_ids"]
        attention_mask = encoded["attention_mask"]

        outputs = await self.engine.generate_batch(input_ids, attention_mask, batch)

        for req, output in zip(batch, outputs):
            req.future.set_result(output)