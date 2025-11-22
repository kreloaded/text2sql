"""
CodeT5+ Inference Service
Generates SQL queries using HuggingFace Inference API or local model.
"""

import requests
import os
from typing import List, Dict, Any


class SQLGenerator:
    def __init__(
        self,
        model_name: str,
        max_length: int = 512,
        use_api: bool = True,
        api_token: str = None,
        custom_api_url: str = None,
    ):
        """
        Initialize the SQL generator.

        Args:
            model_name: HuggingFace model identifier
            max_length: Maximum sequence length
            use_api: Whether to use API (HuggingFace or custom)
            api_token: HuggingFace API token
            custom_api_url: Custom API endpoint (e.g., from Colab)
        """
        self.model_name = model_name
        self.max_length = max_length
        self.use_api = use_api
        self.api_token = api_token or os.getenv("HUGGINGFACE_API_TOKEN")
        self.custom_api_url = custom_api_url or os.getenv("CUSTOM_MODEL_API_URL")

        if self.use_api:
            if self.custom_api_url:
                # Use custom API endpoint (e.g., from Colab)
                self.api_url = f"{self.custom_api_url.rstrip('/')}/generate"
                self.headers = {"Content-Type": "application/json"}
                self.use_custom_api = True
                print(
                    f"✓ SQL generator initialized with custom API: {self.custom_api_url}"
                )
            elif self.api_token:
                # Use HuggingFace Inference API
                self.api_url = (
                    f"https://api-inference.huggingface.co/models/{model_name}"
                )
                self.headers = {"Authorization": f"Bearer {self.api_token}"}
                self.use_custom_api = False
                print(f"✓ SQL generator initialized with HuggingFace Inference API")
            else:
                raise ValueError(
                    "Either CUSTOM_MODEL_API_URL or HUGGINGFACE_API_TOKEN must be set."
                )
        else:
            # Fallback to local model (original code)
            from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
            import torch

            print(f"Loading model and tokenizer from {model_name}...")
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            except Exception as e:
                print(
                    f"Warning: Could not load tokenizer from {model_name}, using Salesforce/codet5-base"
                )
                self.tokenizer = AutoTokenizer.from_pretrained("Salesforce/codet5-base")

            print(f"Loading model from {model_name}...")
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(
                self.device
            )
            self.model.eval()
            print(f"✓ SQL generator initialized on {self.device}")

    def build_prompt(self, question: str, context: List[Dict[str, Any]]) -> str:
        """
        Build the input prompt from question and retrieved context.
        Explicit instructions to prevent copying example patterns.

        Args:
            question: Natural language question
            context: List of retrieved schema/example entries

        Returns:
            Formatted prompt string
        """
        prompt_parts = []

        # Separate target schema from examples
        schemas = [c for c in context if c.get("type") == "schema"]
        examples = [c for c in context if c.get("type") == "example"]

        # Add TARGET database schema
        if schemas:
            target_schema = schemas[0]
            schema_text = target_schema.get("text", "") or target_schema.get(
                "schema", ""
            )
            if schema_text:
                lines = schema_text.split("\n")
                db_line = (
                    [l for l in lines if l.startswith("Database:")][0]
                    if any(l.startswith("Database:") for l in lines)
                    else ""
                )
                table_lines = [l for l in lines if l.strip().startswith("- ")]

                if db_line:
                    prompt_parts.append(db_line)
                if table_lines:
                    prompt_parts.append("Tables:")
                    prompt_parts.extend(table_lines)
                prompt_parts.append("")

        # Add clear instruction separator
        if examples:
            prompt_parts.append(
                "# Reference Examples (for understanding SQL patterns only):"
            )
            prompt_parts.append(
                "# DO NOT copy these queries - generate NEW SQL based on the question below"
            )
            prompt_parts.append("")
            for example in examples[:3]:
                ex_question = example.get("question", "")
                ex_sql = example.get("sql", "")
                if ex_question and ex_sql:
                    prompt_parts.append(f"Q: {ex_question}")
                    prompt_parts.append(f"A: {ex_sql}")
                    prompt_parts.append("")

        # Clear separator before actual question
        prompt_parts.append("# Generate SQL for THIS question using the schema above:")
        prompt_parts.append(f"Q: {question}")
        prompt_parts.append("A:")

        final_prompt = "\n".join(prompt_parts)
        print(
            f"DEBUG: Built prompt with {len(context)} context entries, total length {len(final_prompt)} chars"
        )
        print(f"DEBUG: Prompt preview:\n{final_prompt[:500]}...")
        return final_prompt

    def generate_sql(
        self,
        question: str,
        context: List[Dict[str, Any]] = None,
        max_new_tokens: int = 128,
        num_beams: int = 1,
        temperature: float = 0.7,
    ) -> str:
        """
        Generate SQL query from natural language question.

        Args:
            question: Natural language question
            context: Optional retrieved context (schema/examples)
            max_new_tokens: Maximum tokens to generate
            num_beams: Number of beams for beam search
            temperature: Sampling temperature

        Returns:
            Generated SQL query string
        """
        try:
            # Build prompt with context if available
            if context:
                prompt = self.build_prompt(question, context)
            else:
                prompt = f"Question: {question}\nSQL:"

            print(f"DEBUG: Prompt length: {len(prompt)} chars")
            print(f"DEBUG: Generating SQL with max_new_tokens={max_new_tokens}")
            print(f"DEBUG: Prompt: {prompt}")

            if self.use_api:
                # Use HuggingFace Inference API
                return self._generate_with_api(prompt, max_new_tokens)
            else:
                # Use local model
                return self._generate_with_local_model(
                    prompt, max_new_tokens, num_beams
                )

        except Exception as e:
            print(f"ERROR in generate_sql: {str(e)}")
            import traceback

            traceback.print_exc()
            raise

    def _generate_with_api(self, prompt: str, max_new_tokens: int = 128) -> str:
        """Generate SQL using API (HuggingFace or custom)."""
        try:
            if self.use_custom_api:
                # Custom API format (Colab)
                payload = {"prompt": prompt, "max_new_tokens": max_new_tokens}
            else:
                # HuggingFace API format
                payload = {
                    "inputs": prompt,
                    "parameters": {
                        "max_new_tokens": max_new_tokens,
                        "return_full_text": False,
                        "do_sample": False,
                    },
                    "options": {"wait_for_model": True},
                }

            print(f"DEBUG: Calling HuggingFace API...")
            response = requests.post(
                self.api_url, headers=self.headers, json=payload, timeout=60
            )

            print(f"DEBUG: Status code: {response.status_code}")

            if response.status_code == 200:
                result = response.json()

                if self.use_custom_api:
                    # Custom API response format
                    sql_query = result.get("generated_sql", "")
                else:
                    # HuggingFace API response format
                    if isinstance(result, list) and len(result) > 0:
                        sql_query = result[0].get("generated_text", "")
                    else:
                        sql_query = result.get("generated_text", str(result))

                print(f"DEBUG: Generated SQL: {sql_query[:200]}")
                return sql_query.strip()
            elif response.status_code == 503:
                # Model is loading
                error_msg = (
                    "Model is currently loading. Please try again in a few moments."
                )
                print(f"WARNING: {error_msg}")
                raise Exception(error_msg)
            else:
                error_msg = (
                    f"HuggingFace API error: {response.status_code} - {response.text}"
                )
                print(f"ERROR: {error_msg}")
                raise Exception(error_msg)

        except requests.exceptions.RequestException as e:
            error_msg = f"HuggingFace API request error: {str(e)}"
            print(f"ERROR: {error_msg}")
            raise Exception(error_msg)
        except Exception as e:
            error_msg = f"HuggingFace API error: {str(e)}"
            print(f"ERROR: {error_msg}")
            raise Exception(error_msg)

    def _generate_with_local_model(
        self, prompt: str, max_new_tokens: int = 128, num_beams: int = 1
    ) -> str:
        """Generate SQL using local model."""
        import torch

        print(f"DEBUG: Prompt preview: {prompt[:200]}...")

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            max_length=self.max_length,
            truncation=True,
            padding=True,
        )

        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs.get("attention_mask", None)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)

        print(f"DEBUG: Input tokens shape: {input_ids.shape}")

        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                num_return_sequences=1,
            )

        print(f"DEBUG: Output shape: {outputs.shape}")
        sql_query = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"DEBUG: Generated SQL: {sql_query[:200]}")
        return sql_query.strip()

    def generate_batch(
        self,
        questions: List[str],
        contexts: List[List[Dict[str, Any]]] = None,
        max_new_tokens: int = 256,
        num_beams: int = 4,
    ) -> List[str]:
        """
        Generate SQL queries for a batch of questions.

        Args:
            questions: List of natural language questions
            contexts: Optional list of retrieved contexts
            max_new_tokens: Maximum tokens to generate
            num_beams: Number of beams for beam search

        Returns:
            List of generated SQL query strings
        """
        prompts = []
        for i, question in enumerate(questions):
            if contexts and i < len(contexts):
                prompts.append(self.build_prompt(question, contexts[i]))
            else:
                prompts.append(f"Question: {question}\nSQL:")

        # Tokenize batch
        inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            max_length=self.max_length,
            truncation=True,
            padding=True,
        ).to(self.device)

        # Generate batch
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                num_beams=num_beams,
                early_stopping=True,
            )

        # Decode all
        sql_queries = [
            self.tokenizer.decode(output, skip_special_tokens=True).strip()
            for output in outputs
        ]

        return sql_queries
