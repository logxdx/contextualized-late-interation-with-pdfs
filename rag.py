#################
# Timer Wrapper #
#################

import time
import functools
from contextlib import ContextDecorator


class Timer(ContextDecorator):
    """
    A versatile timer that can be used as:
    - A decorator to time functions.
    - A context manager to time code blocks.

    Example:
        @Timer()
        def slow_func():
            time.sleep(2)

        with Timer("Block Timer"):
            time.sleep(1.5)
    """

    def __init__(self, name=None, logger=print):
        self.name = name
        self.logger = logger
        self.start_time = None
        self.elapsed = None

    def __enter__(self):
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, *exc):
        self.elapsed = time.perf_counter() - self.start_time  # type: ignore
        label = f"[{self.name}] " if self.name else ""
        self.logger(f"{label}Elapsed time: {self.elapsed:.6f} seconds")
        return False  # Don't suppress exceptions

    def __call__(self, func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            with self:
                return func(*args, **kwargs)

        return wrapper


####################
# Late Interaction #
####################

import io
import gc
import os
import base64
from typing import Optional, Union, cast
from uuid import uuid4
from pathlib import Path
from textwrap import dedent

import torch
from colpali_engine.models import (
    ColPali,
    ColPaliProcessor,
    ColQwen2,
    ColQwen2Processor,
    ColQwen2_5,
    ColQwen2_5_Processor,
    ColIdefics3,
    ColIdefics3Processor,
)

import litellm
import numpy as np
from qdrant_client import QdrantClient, models
from pdf2image import convert_from_path
from PIL import Image
from typing import Optional


class ColPaliModel:

    def __init__(
        self,
        pretrained_model_name_or_path: str,
        model_dtype: torch.dtype = torch.bfloat16,
        device: Optional[Union[str, torch.device]] = None,
        **kwargs,
    ):

        if (
            "colpali" not in pretrained_model_name_or_path.lower()
            and "colqwen2" not in pretrained_model_name_or_path.lower()
            and "colsmol" not in pretrained_model_name_or_path.lower()
        ):
            raise ValueError(
                "This module only supports ColPali, ColQwen2, ColQwen2.5, ColSmol-256M & ColSmol-500M for now. Incorrect model name specified."
            )

        self.pretrained_model_name_or_path = pretrained_model_name_or_path
        self.model_dtype = model_dtype
        device = device or (
            "cuda"
            if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available() else "cpu"
        )

        if "colpali" in pretrained_model_name_or_path.lower():
            self.model = ColPali.from_pretrained(
                self.pretrained_model_name_or_path,
                device_map=device,
                torch_dtype=self.model_dtype,
                # token=kwargs.get("hf_token", None) or os.environ.get("HF_TOKEN"),
            )
        elif "colqwen2.5" in pretrained_model_name_or_path.lower():
            self.model = ColQwen2_5.from_pretrained(
                self.pretrained_model_name_or_path,
                device_map=device,
                torch_dtype=self.model_dtype,
                # token=kwargs.get("hf_token", None) or os.environ.get("HF_TOKEN"),
            )
        elif "colqwen2" in pretrained_model_name_or_path.lower():
            self.model = ColQwen2.from_pretrained(
                self.pretrained_model_name_or_path,
                device_map=device,
                torch_dtype=self.model_dtype,
                # token=kwargs.get("hf_token", None) or os.environ.get("HF_TOKEN"),
            )
        elif "colsmol" in pretrained_model_name_or_path.lower():
            self.model = ColIdefics3.from_pretrained(
                self.pretrained_model_name_or_path,
                device_map=device,
                torch_dtype=self.model_dtype,
                # token=kwargs.get("hf_token", None) or os.environ.get("HF_TOKEN"),
            )

        # set to eval mode
        self.model = self.model.eval()

        self.patches = tuple()

        if "colpali" in pretrained_model_name_or_path.lower():
            self.processor = cast(
                ColPaliProcessor,
                ColPaliProcessor.from_pretrained(
                    self.pretrained_model_name_or_path,
                    use_fast=True,
                    # token=kwargs.get("hf_token", None) or os.environ.get("HF_TOKEN"),
                ),
            )
        elif "colqwen2.5" in pretrained_model_name_or_path.lower():
            self.processor = cast(
                ColQwen2_5_Processor,
                ColQwen2_5_Processor.from_pretrained(
                    self.pretrained_model_name_or_path,
                    use_fast=True,
                    # token=kwargs.get("hf_token", None) or os.environ.get("HF_TOKEN"),
                ),
            )
        elif "colqwen2" in pretrained_model_name_or_path.lower():
            self.processor = cast(
                ColQwen2Processor,
                ColQwen2Processor.from_pretrained(
                    self.pretrained_model_name_or_path,
                    use_fast=True,
                    # token=kwargs.get("hf_token", None) or os.environ.get("HF_TOKEN"),
                ),
            )
        elif "colsmol" in pretrained_model_name_or_path.lower():
            self.processor = cast(
                ColIdefics3Processor,
                ColIdefics3Processor.from_pretrained(
                    self.pretrained_model_name_or_path,
                    use_fast=True,
                    # token=kwargs.get("hf_token", None) or os.environ.get("HF_TOKEN"),
                ),
            )

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        device: Optional[Union[str, torch.device]] = None,
        **kwargs,
    ):
        return cls(
            pretrained_model_name_or_path=pretrained_model_name_or_path,
            device=device,
            **kwargs,
        )

    def encode_queries(self, texts: Union[str, list[str]]) -> np.ndarray:
        if isinstance(texts, str):
            texts = [texts]
        with torch.inference_mode():
            batch_text = self.processor.process_texts(texts).to(self.model.device)
            embeddings = self.model(**batch_text).detach().cpu().float().numpy()
        del batch_text
        torch.cuda.empty_cache()
        return embeddings

    def encode_images(
        self, images: Union[Image.Image, list[Image.Image]]
    ) -> np.ndarray:
        if isinstance(images, Image.Image):
            images = [images]
        with torch.inference_mode():
            batch_images = self.processor.process_images(images).to(self.model.device)
            embeddings = self.model(**batch_images).detach().cpu().float().numpy()
        del batch_images
        torch.cuda.empty_cache()
        return embeddings

    def get_patches(self, image_size: tuple[int, int]) -> tuple[int, int]:

        if "colpali" in self.pretrained_model_name_or_path.lower():
            return self.processor.get_n_patches(image_size, patch_size=self.model.patch_size)  # type: ignore
        elif "colqwen" in self.pretrained_model_name_or_path.lower():
            return self.processor.get_n_patches(image_size, spatial_merge_size=self.model.spatial_merge_size)  # type: ignore
        return 0, 0

    @Timer("Batched Pooled Embeddings")
    def batch_pooled_embeddings(self, image_batch: list[Image.Image]):

        # embed
        with torch.inference_mode():
            processed_images = self.processor.process_images(image_batch).to(
                self.model.device
            )
            image_embeddings = self.model(**processed_images)

        tokenized_images = processed_images.input_ids

        del processed_images
        torch.cuda.empty_cache()

        # mean pooling
        pooled_by_rows_batch = []
        pooled_by_columns_batch = []

        for image_embedding, tokenized_image, image in zip(
            image_embeddings, tokenized_images, image_batch
        ):
            x_patches, y_patches = self.get_patches(image.size)

            image_tokens_mask = tokenized_image == self.processor.image_token_id

            # Number of actual image tokens for this sample
            num_image_tokens = image_tokens_mask.sum().item()
            # Derive x/y patch counts safely
            if (
                x_patches <= 0
                or y_patches <= 0
                or (x_patches * y_patches != num_image_tokens)
            ):
                # Default to square-ish layout if metadata is wrong
                y_patches = int((num_image_tokens) ** 0.5)
                x_patches = num_image_tokens // y_patches

            image_tokens = image_embedding[image_tokens_mask].view(
                x_patches, y_patches, self.model.dim
            )
            pooled_by_rows = torch.mean(image_tokens, dim=0)
            pooled_by_columns = torch.mean(image_tokens, dim=1)

            image_token_idxs = torch.nonzero(image_tokens_mask.int(), as_tuple=False)
            first_image_token_idx = image_token_idxs[0].cpu().item()
            last_image_token_idx = image_token_idxs[-1].cpu().item()

            if first_image_token_idx == 0 and last_image_token_idx == len(
                image_embedding - 1
            ):
                pooled_by_rows = pooled_by_rows.cpu().float().numpy().tolist()
                pooled_by_columns = pooled_by_columns.cpu().float().numpy().tolist()
            else:
                prefix_tokens = image_embedding[:first_image_token_idx]
                postfix_tokens = image_embedding[last_image_token_idx + 1 :]

                # adding back prefix and postfix special tokens
                pooled_by_rows = (
                    torch.cat((prefix_tokens, pooled_by_rows, postfix_tokens), dim=0)
                    .cpu()
                    .float()
                    .numpy()
                    .tolist()
                )
                pooled_by_columns = (
                    torch.cat((prefix_tokens, pooled_by_columns, postfix_tokens), dim=0)
                    .cpu()
                    .float()
                    .numpy()
                    .tolist()
                )

            pooled_by_rows_batch.append(pooled_by_rows)
            pooled_by_columns_batch.append(pooled_by_columns)

            del (
                x_patches,
                y_patches,
                image_tokens_mask,
                image_tokens,
                pooled_by_rows,
                pooled_by_columns,
                first_image_token_idx,
                last_image_token_idx,
                prefix_tokens,
                postfix_tokens,
            )

        image_embeddings = image_embeddings.cpu().float().numpy().tolist()

        del tokenized_images
        torch.cuda.empty_cache()
        gc.collect()

        return image_embeddings, pooled_by_rows_batch, pooled_by_columns_batch


class QdrantVectorStore:

    def __init__(
        self, index_path: str = "./index", collection_name: str = "rag"
    ) -> None:

        self.client = QdrantClient(path=index_path)
        self.collection_name = collection_name

        self.vectors_config = {
            "original": models.VectorParams(
                size=128,
                distance=models.Distance.COSINE,
                on_disk=True,
                multivector_config=models.MultiVectorConfig(
                    comparator=models.MultiVectorComparator.MAX_SIM
                ),
                hnsw_config=models.HnswConfigDiff(m=0),
                quantization_config=models.ScalarQuantization(
                    scalar=models.ScalarQuantizationConfig(
                        type=models.ScalarType.INT8,
                        quantile=0.99,
                    )
                ),
            ),
            "mean_pooling_columns": models.VectorParams(
                size=128,
                on_disk=True,
                distance=models.Distance.COSINE,
                multivector_config=models.MultiVectorConfig(
                    comparator=models.MultiVectorComparator.MAX_SIM
                ),
                quantization_config=models.ScalarQuantization(
                    scalar=models.ScalarQuantizationConfig(
                        type=models.ScalarType.INT8,
                        quantile=0.99,
                    )
                ),
            ),
            "mean_pooling_rows": models.VectorParams(
                size=128,
                on_disk=True,
                distance=models.Distance.COSINE,
                multivector_config=models.MultiVectorConfig(
                    comparator=models.MultiVectorComparator.MAX_SIM
                ),
                quantization_config=models.ScalarQuantization(
                    scalar=models.ScalarQuantizationConfig(
                        type=models.ScalarType.INT8,
                        quantile=0.99,
                    )
                ),
            ),
        }

        self.create_collection(collection_name=self.collection_name)

        print(f"Collection {self.collection_name} created")

    def _collection_exists(self, collection_name: Optional[str] = None):
        if not collection_name:
            collection_name = self.collection_name
        return self.client.collection_exists(collection_name=collection_name)

    @Timer("Create Collection")
    def create_collection(self, collection_name: str):
        if not self._collection_exists(collection_name=collection_name):
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config=self.vectors_config,
                on_disk_payload=True,
            )
            return True
        return False

    def delete_collection(self, collection_name: str):
        if self._collection_exists(collection_name=collection_name):
            self.client.delete_collection(collection_name=collection_name)
            return True
        return False

    @Timer("Upload Batch Vectors")
    def upload_batch_vectors(
        self,
        original_batch,
        pooled_by_rows_batch,
        pooled_by_columns_batch,
        payload_batch: list[dict],
    ):

        try:
            self.client.upload_collection(
                collection_name=self.collection_name,
                vectors={
                    "original": original_batch,
                    "mean_pooling_rows": pooled_by_rows_batch,
                    "mean_pooling_columns": pooled_by_columns_batch,
                },
                payload=payload_batch,
                ids=[str(uuid4()) for _ in range(len(original_batch))],
            )
        except Exception as e:
            print(f"Error uploading upsert: {e}")

    @Timer("Vector Search")
    def search(
        self,
        queries: np.ndarray,
        search_limit: int = 10,
        prefetch_limit: int = 100,
        collection_name: Optional[str] = None,
    ):

        if collection_name:
            if self._collection_exists(collection_name=collection_name):
                self.collection_name = collection_name

        search_queries = [
            models.QueryRequest(
                query=query,
                prefetch=[
                    models.Prefetch(
                        query=query,
                        limit=prefetch_limit,
                        using="mean_pooling_rows",
                    ),
                    models.Prefetch(
                        query=query,
                        limit=prefetch_limit,
                        using="mean_pooling_columns",
                    ),
                ],
                limit=search_limit,
                with_payload=True,
                with_vector=False,
                using="original",
            )
            for query in queries
        ]

        response = self.client.query_batch_points(
            requests=search_queries, collection_name=self.collection_name
        )
        return [result.points for result in response]

    def upsert_batch_vectors(
        self,
        ids: list[str],
        originals: list[list[float]],
        pooled_rows: list[list[float]],
        pooled_cols: list[list[float]],
        payloads: list[dict],
    ):
        """
        Upsert a list of points (multivector) to Qdrant using PointStruct.vectors.
        """
        points = []
        for _id, orig, prow, pcol, payload in zip(
            ids, originals, pooled_rows, pooled_cols, payloads
        ):
            pts = models.PointStruct(
                id=_id,
                vector={
                    "original": orig,
                    "mean_pooling_rows": prow,
                    "mean_pooling_columns": pcol,
                },
                payload=payload,
            )
            points.append(pts)

        try:
            # Upsert is incremental and suitable for streaming ingestion
            self.client.upsert(collection_name=self.collection_name, points=points)
        except Exception as e:
            print(f"[Qdrant] Upsert error: {e}")


class RAG:

    def __init__(
        self,
        rag_model: str = "vidore/colSmol-500M",
        model_dtype: torch.dtype = torch.bfloat16,
        device: Optional[Union[str, torch.device]] = None,
        index_path: str = "./index",
        collection_name: str = "rag",
    ) -> None:
        """
        model: Name of the RAG model
        """
        self.model = ColPaliModel(
            pretrained_model_name_or_path=rag_model,
            model_dtype=model_dtype,
            device=device,
        )
        self.vector_store = QdrantVectorStore(
            index_path=index_path, collection_name=collection_name
        )

    @Timer("Index Image Batch")
    def _index_batch(self, image_queue: list):

        image_batch = []
        payload_batch = []
        while image_queue:
            image, payload = image_queue.pop(0)
            image_batch.append(image)
            payload_batch.append(payload)

        original_batch, pooled_by_rows_batch, pooled_by_columns_batch = (
            self.model.batch_pooled_embeddings(image_batch)
        )
        self.vector_store.upload_batch_vectors(
            np.asarray(original_batch),
            np.asarray(pooled_by_rows_batch),
            np.asarray(pooled_by_columns_batch),
            [payload],
        )
        del original_batch, pooled_by_rows_batch, pooled_by_columns_batch

    @Timer("PDF to Image")
    def _pdf_to_image(self, pdf_path: Union[str, Path]):

        if isinstance(pdf_path, str):
            pdf_path = Path(pdf_path).resolve()

        images = convert_from_path(
            pdf_path=str(pdf_path),
            dpi=100,
            thread_count=max(1, int(os.cpu_count() // 2)),  # type: ignore
        )

        for page_no, image in enumerate(images, 1):
            buffer = io.BytesIO()
            image.save(buffer, format="jpeg", quality=75)
            yield image, {
                "file": pdf_path.name,
                "page_no": page_no,
                "image": base64.b64encode(buffer.getvalue()).decode("utf-8"),
            }
            del buffer, image

    @Timer("Index Folder")
    def index_folder(self, path: Union[str, Path], batch_size: int = 1):

        if isinstance(path, str):
            path = Path(path).resolve()

        pdf_files = list(path.glob("*.pdf"))
        image_queue = []
        for pdf_path in pdf_files:
            self.index_file(pdf_path=pdf_path, batch_size=batch_size)

        del image_queue

    @Timer("Index PDF File")
    def index_file(self, pdf_path: Union[str, Path], batch_size: int = 1):

        if isinstance(pdf_path, str):
            pdf_path = Path(pdf_path).resolve()

        image_queue = []
        for image, payload in self._pdf_to_image(pdf_path):
            image_queue.append((image, payload))

            if len(image_queue) >= batch_size:
                self._index_batch(image_queue)

        del image_queue

    @Timer("Answer")
    def answer(
        self,
        query: str,
        top_k: int = 2,
        prefetch_limit: int = 10,
        model_name: str = "gemma3",
        api_key: str = "ollama",
        base_url: str = "http://localhost:11434/v1",
        with_images: bool = False,
    ):
        queries = self.model.encode_queries(query)
        points = self.vector_store.search(
            queries=queries,
            search_limit=top_k,
            prefetch_limit=prefetch_limit,
        )[0]

        system_prompt = dedent(
            f"""
        You are an intelligent assistant that answers user questions only based on the provided context (the images).

        You are given pages from documents. Use its visual and textual information to accurately and concisely answer the user's question. If the answer is not present in the image, clearly state that the answer cannot be found.

        Follow these rules:
        - Base your response only on the image content unless explicitly instructed otherwise.
        - Do not hallucinate or make assumptions beyond the visible data.
        - If the image contains tables, diagrams, or equations, interpret them accurately.
        
        Your job is to provide an accurate and concise answer to the user's question based on the provided context.
        """
        )

        llm_message: list = [
            {
                "role": "system",
                "content": [{"type": "text", "text": system_prompt}],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": query,
                    },
                ],
            },
        ]

        image_msgs: list = []
        images: list[dict] = []

        for point in points:
            encoded_image = point.payload["image"]  # type: ignore
            image_msgs.extend(
                [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{encoded_image}"},
                    },
                ]
            )
            images.append(point.payload)

        user_content = llm_message[1]["content"]
        llm_message[1]["content"] = user_content + image_msgs

        del points, image_msgs, queries

        print(f"Generating Answer...")
        response = (
            litellm.completion(
                model=f"openai/{model_name}",
                api_key=api_key,
                base_url=base_url,
                messages=llm_message,
            )
            .choices[0]  # type: ignore
            .message.content  # type: ignore
        )
        print(f"Answer: {response}")

        if with_images:
            return response, images

        return response

    def close(self):
        """Cleanly close the Qdrant client to avoid shutdown-time ImportError."""
        try:
            if (
                hasattr(self.vector_store, "client")
                and self.vector_store.client is not None
            ):
                self.vector_store.client.close()
        except Exception as e:
            print(f"Error while closing Qdrant client: {e}")


if __name__ == "__main__":

    import os
    from dotenv import load_dotenv

    load_dotenv(".env")

    rag = RAG("vidore/colpali-v1.3")
    # rag.index_file(pdf_path=Path("attention_is_all_you_need.pdf"), batch_size=1)

    rag.answer(
        query="How does multi headed attention work?",
        top_k=4,
        prefetch_limit=10,
        api_key=os.getenv("POLLINATIONS_API_KEY"),  # type: ignore
        base_url=os.getenv("POLLINATIONS_BASE_URL"),  # type: ignore
        model_name="openai-fast",
    )

    rag.close()
