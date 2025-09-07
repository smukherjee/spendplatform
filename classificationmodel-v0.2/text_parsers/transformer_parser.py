"""
Transformer-based Text Parser Module
Provides advanced text preprocessing using transformer models (BERT, RoBERTa)
"""

import re
from typing import List, Dict, Any, Optional, Union
import numpy as np

from .base_parser import BaseTextParser


class TransformerTextParser(BaseTextParser):
    """Advanced text parser using transformer models for superior embeddings"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)

        # Model configuration
        self.model_name = self.config.get('model_name', 'bert-base-uncased')
        self.max_length = self.config.get('max_length', 128)
        self.use_gpu = self.config.get('use_gpu', False)
        self.batch_size = self.config.get('batch_size', 16)

        # Initialize transformer model and tokenizer (lazy loading)
        self.model = None
        self.tokenizer = None
        self._model_loaded = False

    def _load_model(self):
        """Load the transformer model and tokenizer"""
        try:
            from transformers import AutoTokenizer, AutoModel
            import torch

            print(f"🔄 Loading transformer model: {self.model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModel.from_pretrained(self.model_name)

            # Move to GPU if available and requested
            if self.use_gpu and torch.cuda.is_available():
                self.model = self.model.cuda()
                print("✅ Model moved to GPU")
            elif self.use_gpu and not torch.cuda.is_available():
                print("⚠️ GPU requested but not available, using CPU")

            print(f"✅ Transformer model '{self.model_name}' loaded successfully")
            self._model_loaded = True

        except ImportError as e:
            raise ImportError(
                f"Transformers library not installed. Install with: pip install transformers torch\n"
                f"Error: {e}"
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load transformer model '{self.model_name}': {e}")

    def preprocess_text(self, text: str) -> str:
        """Basic text preprocessing for transformer models"""
        if not isinstance(text, str) or text == '':
            return ''

        # Convert to lowercase
        text = text.lower()

        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()

        # For transformers, we keep more of the original text structure
        # as the model will handle tokenization
        return text

    def extract_pos_features(self, texts: List[str]) -> List[Dict[str, Any]]:
        """Extract basic features (transformers handle advanced features internally)"""
        features = []

        for text in texts:
            if not text or text == '':
                features.append({
                    'text_length': 0,
                    'word_count': 0,
                    'has_numbers': 0,
                    'has_symbols': 0
                })
                continue

            features.append({
                'text_length': len(text),
                'word_count': len(text.split()),
                'has_numbers': 1 if re.search(r'\d', text) else 0,
                'has_symbols': 1 if re.search(r'[^\w\s]', text) else 0
            })

        return features

    def get_embeddings(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings for the input texts using the transformer model"""
        if not self._model_loaded:
            self._load_model()

        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Transformer model not loaded")

        import torch
        from torch.utils.data import DataLoader, TensorDataset

        # Tokenize texts
        encodings = self.tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=self.max_length,
            return_tensors='pt'
        )

        # Move to device
        device = next(self.model.parameters()).device
        input_ids = encodings['input_ids'].to(device)
        attention_mask = encodings['attention_mask'].to(device)

        # Generate embeddings
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            # Use mean pooling over token embeddings
            embeddings = outputs.last_hidden_state.mean(dim=1)

        return embeddings.cpu().numpy()

    def get_tokenizer_info(self) -> Dict[str, Any]:
        """Get information about the tokenizer"""
        if self.tokenizer is None:
            return {}

        return {
            'vocab_size': self.tokenizer.vocab_size,
            'max_model_input_sizes': self.tokenizer.max_model_input_sizes,
            'model_max_length': self.tokenizer.model_max_length
        }

    def get_parser_info(self) -> Dict[str, Any]:
        """Get detailed information about this transformer parser"""
        info = super().get_parser_info()
        info.update({
            'model_name': self.model_name,
            'max_length': self.max_length,
            'use_gpu': self.use_gpu,
            'batch_size': self.batch_size,
            'tokenizer_info': self.get_tokenizer_info()
        })
        return info


class BERTTextParser(TransformerTextParser):
    """BERT-based text parser"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        if config is None:
            config = {}
        config['model_name'] = config.get('model_name', 'bert-base-uncased')
        super().__init__(config)


class RoBERTaTextParser(TransformerTextParser):
    """RoBERTa-based text parser"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        if config is None:
            config = {}
        config['model_name'] = config.get('model_name', 'roberta-base')
        super().__init__(config)


class DistilBERTTextParser(TransformerTextParser):
    """DistilBERT-based text parser (lighter, faster)"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        if config is None:
            config = {}
        config['model_name'] = config.get('model_name', 'distilbert-base-uncased')
        super().__init__(config)


class LayoutLMv2TextParser(TransformerTextParser):
    """LayoutLMv2-based text parser for document layout understanding"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        if config is None:
            config = {}
        config['model_name'] = config.get('model_name', 'microsoft/layoutlmv2-base-uncased')
        super().__init__(config)

    def _load_model(self):
        """Load LayoutLMv2 model with layout-specific configuration"""
        try:
            from transformers import LayoutLMv2Tokenizer, LayoutLMv2Model
            import torch

            print(f"🔄 Loading LayoutLMv2 model: {self.model_name}")
            self.tokenizer = LayoutLMv2Tokenizer.from_pretrained(self.model_name)
            self.model = LayoutLMv2Model.from_pretrained(self.model_name)

            # Move to GPU if available and requested
            if self.use_gpu and torch.cuda.is_available():
                self.model = self.model.cuda()
                print("✅ Model moved to GPU")
            elif self.use_gpu and not torch.cuda.is_available():
                print("⚠️ GPU requested but not available, using CPU")

            print(f"✅ LayoutLMv2 model '{self.model_name}' loaded successfully")
            self._model_loaded = True

        except ImportError as e:
            raise ImportError(
                f"LayoutLMv2 dependencies not installed. Install with: pip install transformers torch\n"
                f"Error: {e}"
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load LayoutLMv2 model '{self.model_name}': {e}")

    def preprocess_text(self, text: str) -> str:
        """Enhanced preprocessing for layout-aware parsing"""
        if not isinstance(text, str) or text == '':
            return ''

        # Convert to lowercase
        text = text.lower()

        # Remove extra whitespace while preserving some structure
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()

        # For LayoutLMv2, we can preserve more structural information
        return text

    def get_embeddings(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings using LayoutLMv2 with simulated layout information"""
        if not self._model_loaded:
            self._load_model()

        if self.model is None or self.tokenizer is None:
            raise RuntimeError("LayoutLMv2 model not loaded")

        import torch
        from torch.utils.data import DataLoader, TensorDataset

        # For text-only inputs, create dummy bounding boxes
        # In a real scenario, you'd extract actual layout information
        embeddings_list = []

        for text in texts:
            # Tokenize with dummy layout information
            encoding = self.tokenizer(
                text,
                truncation=True,
                padding=True,
                max_length=self.max_length,
                return_tensors='pt'
            )

            # Add dummy bounding boxes (full page for each token)
            # In practice, you'd extract real bounding boxes from documents
            num_tokens = encoding['input_ids'].shape[1]
            dummy_boxes = [[0, 0, 1000, 1000]] * num_tokens  # Full page dummy boxes

            encoding['bbox'] = torch.tensor([dummy_boxes])
            encoding['image'] = torch.zeros(3, 224, 224)  # Dummy image

            # Move to device
            device = next(self.model.parameters()).device
            input_ids = encoding['input_ids'].to(device)
            attention_mask = encoding['attention_mask'].to(device)
            bbox = encoding['bbox'].to(device)
            image = encoding['image'].unsqueeze(0).to(device)

            # Generate embeddings
            with torch.no_grad():
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    bbox=bbox,
                    image=image
                )
                # Use mean pooling over token embeddings
                embeddings = outputs.last_hidden_state.mean(dim=1)

            embeddings_list.append(embeddings.cpu().numpy())

        return np.vstack(embeddings_list)


class DONUTTextParser(TransformerTextParser):
    """DONUT-based text parser for document understanding"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        if config is None:
            config = {}
        config['model_name'] = config.get('model_name', 'naver-clova-ix/donut-base')
        super().__init__(config)

    def _load_model(self):
        """Load DONUT model with document understanding configuration"""
        try:
            from transformers import DonutProcessor, VisionEncoderDecoderModel
            import torch

            print(f"🔄 Loading DONUT model: {self.model_name}")
            self.processor = DonutProcessor.from_pretrained(self.model_name)
            self.model = VisionEncoderDecoderModel.from_pretrained(self.model_name)

            # Move to GPU if available and requested
            if self.use_gpu and torch.cuda.is_available():
                self.model = self.model.cuda()
                print("✅ Model moved to GPU")
            elif self.use_gpu and not torch.cuda.is_available():
                print("⚠️ GPU requested but not available, using CPU")

            print(f"✅ DONUT model '{self.model_name}' loaded successfully")
            self._model_loaded = True

        except ImportError as e:
            raise ImportError(
                f"DONUT dependencies not installed. Install with: pip install transformers torch\n"
                f"Error: {e}"
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load DONUT model '{self.model_name}': {e}")

    def preprocess_text(self, text: str) -> str:
        """Preprocessing optimized for DONUT's document understanding"""
        if not isinstance(text, str) or text == '':
            return ''

        # DONUT works best with structured text
        text = text.lower()
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()

        return text

    def get_embeddings(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings using DONUT with text-to-image simulation"""
        if not self._model_loaded:
            self._load_model()

        if self.model is None or self.processor is None:
            raise RuntimeError("DONUT model not loaded")

        import torch
        from PIL import Image, ImageDraw, ImageFont
        import numpy as np

        embeddings_list = []

        for text in texts:
            try:
                # Create a simple text image for DONUT
                # In practice, you'd use actual document images
                img = Image.new('RGB', (800, 600), color='white')
                draw = ImageDraw.Draw(img)

                # Simple text rendering (you might want to use a proper font)
                try:
                    font = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", 20)
                except:
                    font = ImageFont.load_default()

                # Draw text on image
                y_position = 50
                for line in text.split('\n')[:10]:  # Limit lines
                    draw.text((50, y_position), line, fill='black', font=font)
                    y_position += 30

                # Process image with DONUT
                pixel_values = self.processor(img, return_tensors="pt").pixel_values

                # Move to device
                device = next(self.model.parameters()).device
                pixel_values = pixel_values.to(device)

                # Generate embeddings from encoder
                with torch.no_grad():
                    outputs = self.model.encoder(pixel_values)
                    # Use mean pooling of the last hidden state
                    embeddings = outputs.last_hidden_state.mean(dim=1)

                embeddings_list.append(embeddings.cpu().numpy())

            except Exception as e:
                print(f"⚠️ Error processing text with DONUT: {e}")
                # Fallback to zero embeddings
                embeddings_list.append(np.zeros((1, self.model.config.d_model)))

        return np.vstack(embeddings_list)

    def get_parser_info(self) -> Dict[str, Any]:
        """Get detailed information about this DONUT parser"""
        info = super().get_parser_info()
        info.update({
            'model_type': 'DONUT',
            'supports_images': True,
            'architecture': 'Vision-Encoder-Decoder'
        })
        return info
