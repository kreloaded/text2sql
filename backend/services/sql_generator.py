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
        Focus on generating SIMPLE, CORRECT SQL without overcomplicating.
        """
        prompt_parts = []

        # Separate target schema from examples
        schemas = [c for c in context if c.get("type") == "schema"]
        examples = [c for c in context if c.get("type") == "example"]

        # Add TARGET database schema FIRST (most important)
        if schemas:
            target_schema = schemas[0]
            schema_text = (
                target_schema.get("text", "") or target_schema.get("schema", "") or ""
            )
            if schema_text:
                prompt_parts.append("# Database Schema:")
                prompt_parts.append(schema_text)
                prompt_parts.append("")

        # Add examples ONLY if they're relevant (limit to 2-3 most relevant)
        if examples:
            prompt_parts.append("# Similar Example Queries:")
            prompt_parts.append(
                "# NOTE: These are just examples. Generate SIMPLE, DIRECT SQL for the question below."
            )
            prompt_parts.append(
                "# Do NOT add unnecessary GROUP BY, ORDER BY, or LIMIT unless the question asks for it."
            )
            prompt_parts.append("")

            # Sort by relevance (distance) and limit to top 2
            examples_sorted = sorted(examples, key=lambda x: x.get("distance", 999))[:2]

            for example in examples_sorted:
                ex_question = example.get("question", "")
                ex_sql = example.get("sql", "")
                if ex_question and ex_sql:
                    prompt_parts.append(f"Example Q: {ex_question}")
                    prompt_parts.append(f"Example SQL: {ex_sql}")
                    prompt_parts.append("")

        # Add explicit instructions for the target question
        prompt_parts.append("# Instructions:")
        prompt_parts.append("# 1. Write SIMPLE SQL that directly answers the question")
        prompt_parts.append(
            "# 2. Use exact equality (=) for 'is' questions, not >= or <="
        )
        prompt_parts.append(
            "# 3. Only add GROUP BY if question asks 'for each' or 'by category'"
        )
        prompt_parts.append(
            "# 4. Only add ORDER BY if question asks for 'sorted', 'ordered', 'top', 'highest', 'lowest'"
        )
        prompt_parts.append(
            "# 5. Only add LIMIT if question asks for 'first N', 'top N', or specific count"
        )
        prompt_parts.append("")
        prompt_parts.append("# Now generate SQL for THIS question:")
        prompt_parts.append(f"Question: {question}")
        prompt_parts.append("SQL:")

        final_prompt = "\n".join(prompt_parts)
        return final_prompt

    def _clean_generated_sql(self, sql: str, question: str) -> str:
        """
        Post-process generated SQL to fix common mistakes.

        Args:
            sql: Generated SQL query
            question: Original question

        Returns:
            Cleaned SQL query
        """
        sql = sql.strip()

        # Fix common typos
        sql = sql.replace("openning_year", "opening_year")

        # If question asks for simple count and SQL has unnecessary complexity
        question_lower = question.lower()
        if "count" in question_lower and "how many" in question_lower:
            # Check if SQL has unnecessary GROUP BY, ORDER BY, LIMIT
            if (
                "GROUP BY" in sql.upper()
                and "for each" not in question_lower
                and "by" not in question_lower
            ):
                # Remove unnecessary GROUP BY clauses for simple counts
                import re

                # Remove GROUP BY ... ORDER BY ... LIMIT patterns
                sql = re.sub(r"\s+GROUP BY[^;]+", "", sql, flags=re.IGNORECASE)
                sql = re.sub(r"\s+ORDER BY[^;]+", "", sql, flags=re.IGNORECASE)
                sql = re.sub(r"\s+LIMIT\s+\d+", "", sql, flags=re.IGNORECASE)

        # Fix >= when question says "is" (exact match)
        if " is " in question_lower and ">=" in sql:
            # Replace >= with = for "is" questions
            import re

            sql = re.sub(r">=\s*(\d+)", r"= \1", sql)

        return sql.strip()

    def generate_sql(
        self,
        question: str,
        context: List[Dict[str, Any]] = None,
        max_new_tokens: int = 128,
        num_beams: int = 1,
        temperature: float = 0.7,
    ) -> str:
        """Generate SQL query from natural language question."""
        try:
            # Build prompt with context if available
            if context:
                prompt = self.build_prompt(question, context)
            else:
                prompt = f"Question: {question}\nSQL:"

            print(f"DEBUG: Generating SQL with max_new_tokens={max_new_tokens}")

            if self.use_api:
                sql_query = self._generate_with_api(prompt, max_new_tokens)
            else:
                sql_query = self._generate_with_local_model(
                    prompt, max_new_tokens, num_beams
                )

            # Clean the generated SQL
            sql_query = self._clean_generated_sql(sql_query, question)

            print(f"DEBUG: Final cleaned SQL: {sql_query}")
            return sql_query

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
