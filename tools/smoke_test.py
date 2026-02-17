import sys
import os

# Add repo root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Mock missing dependencies
from unittest.mock import MagicMock
sys.modules["omegaconf"] = MagicMock()
sys.modules["torch"] = MagicMock()
sys.modules["transformers"] = MagicMock()
sys.modules["huggingface_hub"] = MagicMock()
sys.modules["triton"] = MagicMock()


try:
    print("Attempting to import nodes...")
    import nodes
    print("Successfully imported nodes.")

    print("Checking Node Class Mappings...")
    if "INT8_Hunyuan_PromptEnhancer" in nodes.NODE_CLASS_MAPPINGS:
        print("INT8_Hunyuan_PromptEnhancer found in NODE_CLASS_MAPPINGS.")
    else:
        print("ERROR: INT8_Hunyuan_PromptEnhancer NOT found in NODE_CLASS_MAPPINGS.")
        sys.exit(1)

    print("Instantiating Node Class...")
    node_cls = nodes.NODE_CLASS_MAPPINGS["INT8_Hunyuan_PromptEnhancer"]
    node_instance = node_cls()
    print("Successfully instantiated node class.")
    
    print("Smoke test passed!")

except ImportError as e:
    print(f"ImportError: {e}")
    sys.exit(1)
except Exception as e:
    print(f"An error occurred: {e}")
    sys.exit(1)
