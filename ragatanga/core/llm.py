"""
LLM providers module that abstracts different LLM backends.

This module allows for using different language model providers for
query generation and answer generation tasks.
"""

import os
import abc
import json
from typing import Any, Dict, List, Optional, TypeVar, Generic, Type

T = TypeVar('T')

class LLMProvider(abc.ABC):
    """Abstract base class for LLM providers."""
    
    @abc.abstractmethod
    async def generate_text(self,
                           prompt: str,
                           system_prompt: Optional[str] = None,
                           temperature: float = 0.7,
                           max_tokens: int = 1000) -> str:
        """
        Generate text using the LLM.
        
        Args:
            prompt: The user prompt
            system_prompt: Optional system prompt
            temperature: Sampling temperature (0.0 to 1.0)
            max_tokens: Maximum tokens to generate
            
        Returns:
            Generated text
        """
        pass
    
    @abc.abstractmethod
    async def generate_structured(self,
                                 prompt: str,
                                 response_model: Type[T],
                                 system_prompt: Optional[str] = None,
                                 temperature: float = 0.7,
                                 max_tokens: int = 1000) -> T:
        """
        Generate structured output using the LLM.
        
        Args:
            prompt: The user prompt
            response_model: Pydantic model class for the response
            system_prompt: Optional system prompt
            temperature: Sampling temperature (0.0 to 1.0)
            max_tokens: Maximum tokens to generate
            
        Returns:
            Instance of response_model
        """
        pass
    
    @staticmethod
    def get_provider(provider_name: str = None, **kwargs) -> "LLMProvider":
        """
        Factory method to get an LLM provider based on configuration.
        
        Args:
            provider_name: Name of the provider (openai, huggingface, ollama)
            **kwargs: Additional configuration parameters
            
        Returns:
            An instance of LLMProvider
        """
        # Default to environment variable or fallback to OpenAI
        if provider_name is None:
            provider_name = os.getenv("LLM_PROVIDER", "openai")
            
        if provider_name == "openai":
            return OpenAIProvider(**kwargs)
        elif provider_name == "huggingface":
            return HuggingFaceProvider(**kwargs)
        elif provider_name == "ollama":
            return OllamaProvider(**kwargs)
        elif provider_name == "anthropic":
            return AnthropicProvider(**kwargs)
        else:
            raise ValueError(f"Unsupported LLM provider: {provider_name}")

class OpenAIProvider(LLMProvider):
    """OpenAI-based LLM provider."""
    
    def __init__(self, model: str = "gpt-4o", api_key: str = None, **kwargs):
        """
        Initialize the OpenAI provider.
        
        Args:
            model: The OpenAI model to use
            api_key: OpenAI API key (if None, uses OPENAI_API_KEY environment variable)
            **kwargs: Additional parameters for the OpenAI client
        """
        try:
            # Only import if this provider is used
            import openai
            import instructor
            from openai import OpenAI
        except ImportError:
            raise ImportError(
                "OpenAI and instructor packages are required. "
                "Install them with 'pip install openai instructor'"
            )
            
        self.model = model
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY is not set")
            
        # Initialize OpenAI client
        self.client = OpenAI(api_key=self.api_key)
        
        # Initialize instructor client
        self.instructor_client = instructor.from_openai(self.client)
    
    async def generate_text(self,
                           prompt: str,
                           system_prompt: Optional[str] = None,
                           temperature: float = 0.7,
                           max_tokens: int = 1000) -> str:
        """Generate text using OpenAI."""
        import asyncio
        
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        response = await asyncio.to_thread(
            self.client.chat.completions.create,
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens
        )
        
        return response.choices[0].message.content
    
    async def generate_structured(self,
                                 prompt: str,
                                 response_model: Type[T],
                                 system_prompt: Optional[str] = None,
                                 temperature: float = 0.7,
                                 max_tokens: int = 1000) -> T:
        """Generate structured output using OpenAI and instructor."""
        import asyncio
        
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        # Use instructor to get structured output
        response = await asyncio.to_thread(
            self.instructor_client.create,
            response_model=response_model,
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            max_retries=3
        )
        
        return response

class HuggingFaceProvider(LLMProvider):
    """HuggingFace-based LLM provider."""
    
    def __init__(self,
                model: str = "mistralai/Mistral-7B-Instruct-v0.2",
                api_key: str = None,
                api_url: Optional[str] = None,
                local: bool = False,
                **kwargs):
        """
        Initialize the HuggingFace provider.
        
        Args:
            model: The HuggingFace model to use
            api_key: HuggingFace API key for API usage
            api_url: HuggingFace API URL (if None, uses the default)
            local: Whether to use a local model
            **kwargs: Additional parameters
        """
        self.model_name = model
        self.api_key = api_key or os.getenv("HF_API_KEY")
        self.api_url = api_url
        self.local = local
        
        # Additional parameters
        self.device = kwargs.get("device", "cuda" if self._is_cuda_available() else "cpu")
        self.max_length = kwargs.get("max_length", 4096)
        
        # Initialize model and tokenizer if using local models
        if local:
            try:
                from transformers import AutoModelForCausalLM, AutoTokenizer
                import torch
                
                self.tokenizer = AutoTokenizer.from_pretrained(model, use_fast=True)
                self.model = AutoModelForCausalLM.from_pretrained(
                    model,
                    device_map=self.device,
                    torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
                )
            except ImportError:
                raise ImportError(
                    "transformers and torch are required for local HuggingFace models. "
                    "Install them with 'pip install transformers torch'"
                )
    
    def _is_cuda_available(self) -> bool:
        """Check if CUDA is available."""
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False
    
    def _format_prompt(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """Format prompt for the model."""
        if system_prompt:
            return f"<s>[INST] {system_prompt}\n\n{prompt} [/INST]"
        else:
            return f"<s>[INST] {prompt} [/INST]"
    
    async def generate_text(self,
                           prompt: str,
                           system_prompt: Optional[str] = None,
                           temperature: float = 0.7,
                           max_tokens: int = 1000) -> str:
        """Generate text using HuggingFace."""
        import asyncio
        
        if self.local:
            # Use local model
            formatted_prompt = self._format_prompt(prompt, system_prompt)
            
            def _generate():
                import torch
                
                with torch.no_grad():
                    inputs = self.tokenizer(formatted_prompt, return_tensors="pt")
                    inputs = {k: v.to(self.device) for k, v in inputs.items()}
                    
                    # Generate
                    outputs = self.model.generate(
                        **inputs,
                        max_length=len(inputs["input_ids"][0]) + max_tokens,
                        temperature=temperature,
                        do_sample=temperature > 0.0,
                        pad_token_id=self.tokenizer.eos_token_id
                    )
                    
                    # Decode and extract only the response part
                    full_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                    
                    # Handle different model output formats
                    if "[/INST]" in full_text:
                        # Mistral and similar format
                        response = full_text.split("[/INST]", 1)[1].strip()
                    else:
                        # Fallback: just return everything after the prompt
                        response = full_text[len(formatted_prompt):].strip()
                    
                    return response
            
            return await asyncio.to_thread(_generate)
        else:
            # Use HuggingFace API
            try:
                import requests
                
                # Format messages for the API
                messages = []
                if system_prompt:
                    messages.append({"role": "system", "content": system_prompt})
                messages.append({"role": "user", "content": prompt})
                
                headers = {
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                }
                
                payload = {
                    "model": self.model_name,
                    "messages": messages,
                    "temperature": temperature,
                    "max_tokens": max_tokens
                }
                
                api_url = self.api_url or f"https://api-inference.huggingface.co/models/{self.model_name}"
                
                def _call_api():
                    response = requests.post(api_url, headers=headers, json=payload)
                    response.raise_for_status()
                    return response.json()["generated_text"]
                
                return await asyncio.to_thread(_call_api)
                
            except ImportError:
                raise ImportError("requests is required for HuggingFace API calls")
    
    async def generate_structured(self,
                                 prompt: str,
                                 response_model: Type[T],
                                 system_prompt: Optional[str] = None,
                                 temperature: float = 0.7,
                                 max_tokens: int = 1000) -> T:
        """Generate structured output using HuggingFace."""
        import json
        from pydantic import ValidationError
        
        # First, generate a structured prompt that specifies the expected format
        schema = response_model.schema()
        schema_json = json.dumps(schema, indent=2)
        
        structured_system_prompt = f"""
        You must respond with a valid JSON object that conforms to the following schema:
        
        {schema_json}
        
        Your response must be valid JSON without any explanations or markdown.
        """
        
        # Combine with the original system prompt if provided
        combined_system_prompt = f"{system_prompt}\n\n{structured_system_prompt}" if system_prompt else structured_system_prompt
        
        # Generate the text
        json_text = await self.generate_text(
            prompt=prompt,
            system_prompt=combined_system_prompt,
            temperature=temperature,
            max_tokens=max_tokens
        )
        
        # Extract JSON part (in case there's any extra text)
        try:
            # Try to find JSON by looking for opening brace
            start_idx = json_text.find("{")
            end_idx = json_text.rfind("}")
            
            if start_idx >= 0 and end_idx >= 0:
                json_text = json_text[start_idx:end_idx+1]
            
            # Parse JSON and validate
            data = json.loads(json_text)
            return response_model.parse_obj(data)
        except (json.JSONDecodeError, ValidationError) as e:
            # If JSON parsing fails, retry with more explicit instructions
            retry_prompt = f"""
            Your previous response was not valid JSON. Please provide a valid JSON object
            matching this schema: {schema_json}
            
            Original query: {prompt}
            
            DO NOT include explanation text, ONLY include a valid JSON object.
            """
            
            json_text = await self.generate_text(
                prompt=retry_prompt,
                system_prompt=None,  # Already included in the retry prompt
                temperature=temperature,
                max_tokens=max_tokens
            )
            
            # Try again
            try:
                start_idx = json_text.find("{")
                end_idx = json_text.rfind("}")
                
                if start_idx >= 0 and end_idx >= 0:
                    json_text = json_text[start_idx:end_idx+1]
                
                data = json.loads(json_text)
                return response_model.parse_obj(data)
            except (json.JSONDecodeError, ValidationError) as e:
                raise ValueError(f"Failed to generate valid structured output: {str(e)}")

class OllamaProvider(LLMProvider):
    """Ollama-based LLM provider for local deployment."""
    
    def __init__(self,
                model: str = "llama3",
                api_url: str = "http://localhost:11434",
                **kwargs):
        """
        Initialize the Ollama provider.
        
        Args:
            model: The Ollama model to use
            api_url: URL of the Ollama API
            **kwargs: Additional parameters
        """
        self.model = model
        self.api_url = api_url
    
    async def generate_text(self,
                           prompt: str,
                           system_prompt: Optional[str] = None,
                           temperature: float = 0.7,
                           max_tokens: int = 1000) -> str:
        """Generate text using Ollama."""
        import json
        import aiohttp
        
        async with aiohttp.ClientSession() as session:
            payload = {
                "model": self.model,
                "prompt": prompt,
                "temperature": temperature,
                "max_tokens": max_tokens
            }
            
            if system_prompt:
                payload["system"] = system_prompt
            
            async with session.post(f"{self.api_url}/api/generate", json=payload) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise ValueError(f"Ollama API error: {error_text}")
                
                response_text = await response.text()
                response_json = json.loads(response_text)
                return response_json.get("response", "")
    
    async def generate_structured(self,
                                 prompt: str,
                                 response_model: Type[T],
                                 system_prompt: Optional[str] = None,
                                 temperature: float = 0.7,
                                 max_tokens: int = 1000) -> T:
        """Generate structured output using Ollama."""
        import json
        from pydantic import ValidationError
        
        # Create a prompt that asks for JSON with the required schema
        schema = response_model.schema()
        schema_json = json.dumps(schema, indent=2)
        
        structured_prompt = f"""
        You must respond with a valid JSON object that conforms to the following schema:
        
        {schema_json}
        
        Original request: {prompt}
        
        Your response must be valid JSON without any explanations or markdown.
        """
        
        combined_system_prompt = f"""
        You are a structured data generation assistant.
        Always respond with valid JSON that matches the requested schema exactly.
        {system_prompt or ''}
        """
        
        # Generate the text
        json_text = await self.generate_text(
            prompt=structured_prompt,
            system_prompt=combined_system_prompt,
            temperature=temperature,
            max_tokens=max_tokens
        )
        
        # Extract JSON part (in case there's any extra text)
        try:
            # Try to find JSON by looking for opening brace
            start_idx = json_text.find("{")
            end_idx = json_text.rfind("}")
            
            if start_idx >= 0 and end_idx >= 0:
                json_text = json_text[start_idx:end_idx+1]
            
            # Parse JSON and validate
            data = json.loads(json_text)
            return response_model.parse_obj(data)
        except (json.JSONDecodeError, ValidationError) as e:
            # If JSON parsing fails, retry with more explicit instructions
            retry_prompt = f"""
            Your previous response was not valid JSON. Please provide a valid JSON object
            matching this schema: {schema_json}
            
            Original query: {prompt}
            
            DO NOT include explanation text, ONLY include a valid JSON object.
            """
            
            json_text = await self.generate_text(
                prompt=retry_prompt,
                system_prompt=None,
                temperature=temperature,
                max_tokens=max_tokens
            )
            
            # Try again
            try:
                start_idx = json_text.find("{")
                end_idx = json_text.rfind("}")
                
                if start_idx >= 0 and end_idx >= 0:
                    json_text = json_text[start_idx:end_idx+1]
                
                data = json.loads(json_text)
                return response_model.parse_obj(data)
            except (json.JSONDecodeError, ValidationError) as e:
                raise ValueError(f"Failed to generate valid structured output: {str(e)}")

class AnthropicProvider(LLMProvider):
    """Anthropic Claude-based LLM provider."""
    
    def __init__(self, model: str = "claude-3-opus-20240229", api_key: str = None, **kwargs):
        """
        Initialize the Anthropic provider.
        
        Args:
            model: The Anthropic model to use
            api_key: Anthropic API key (if None, uses ANTHROPIC_API_KEY environment variable)
            **kwargs: Additional parameters
        """
        try:
            import anthropic
        except ImportError:
            raise ImportError(
                "Anthropic package is required. "
                "Install it with 'pip install anthropic'"
            )
            
        self.model = model
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError("ANTHROPIC_API_KEY is not set")
            
        # Initialize Anthropic client
        self.client = anthropic.Anthropic(api_key=self.api_key)
    
    async def generate_text(self,
                           prompt: str,
                           system_prompt: Optional[str] = None,
                           temperature: float = 0.7,
                           max_tokens: int = 1000) -> str:
        """Generate text using Anthropic Claude."""
        import asyncio
        import anthropic
        
        def _generate():
            system = system_prompt or "You are a helpful assistant."
            message = self.client.messages.create(
                model=self.model,
                system=system,
                max_tokens=max_tokens,
                temperature=temperature,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            return message.content[0].text
        
        return await asyncio.to_thread(_generate)
    
    async def generate_structured(self,
                                 prompt: str,
                                 response_model: Type[T],
                                 system_prompt: Optional[str] = None,
                                 temperature: float = 0.7,
                                 max_tokens: int = 1000) -> T:
        """Generate structured output using Anthropic Claude."""
        import json
        from pydantic import ValidationError
        import asyncio
        import anthropic
        
        # Create a schema string and tool to use with Claude
        schema = response_model.schema()
        
        # Define the Claude tool for JSON output
        json_schema_tool = {
            "name": "extract_structured_data",
            "description": "Extract structured data according to the provided schema",
            "input_schema": schema
        }
        
        # Set up the system prompt
        system = system_prompt or "You are a helpful assistant specializing in structured data extraction."
        
        def _generate_structured():
            message = self.client.messages.create(
                model=self.model,
                system=system,
                max_tokens=max_tokens,
                temperature=temperature,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                tools=[json_schema_tool]
            )
            
            # Look for tool calls in the response
            for content in message.content:
                if content.type == "tool_call" and content.name == "extract_structured_data":
                    return content.input
            
            # If no tool call, try to parse JSON from text
            for content in message.content:
                if content.type == "text":
                    # Look for JSON in the text
                    text = content.text
                    try:
                        # Try to find JSON by looking for opening brace
                        start_idx = text.find("{")
                        end_idx = text.rfind("}")
                        
                        if start_idx >= 0 and end_idx >= 0:
                            json_text = text[start_idx:end_idx+1]
                            data = json.loads(json_text)
                            return data
                    except json.JSONDecodeError:
                        pass
            
            raise ValueError("Failed to extract structured data from Claude response")
        
        try:
            data = await asyncio.to_thread(_generate_structured)
            return response_model.parse_obj(data)
        except (json.JSONDecodeError, ValidationError) as e:
            raise ValueError(f"Failed to generate valid structured output: {str(e)}")