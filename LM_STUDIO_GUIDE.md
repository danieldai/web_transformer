# LM Studio Setup Guide

Complete guide for using your fine-tuned Qwen classifier in LM Studio on macOS M1.

## Prerequisites

- macOS with M1/M2/M3 chip
- LM Studio installed (download from https://lmstudio.ai)
- Your trained model exported to GGUF format

## Step 1: Export Model to GGUF Format

If you haven't already exported your model:

### On Linux (after training):
```bash
source venv/bin/activate
python export_to_gguf.py
```

This creates:
```
qwen_classifier_gguf/
├── qwen-classifier-f16.gguf      # ~500MB (highest quality)
├── qwen-classifier-q4_k_m.gguf   # ~150-200MB (recommended)
├── qwen-classifier-q5_k_m.gguf   # ~200-250MB (higher quality)
└── qwen-classifier-q8_0.gguf     # ~300-350MB (near-lossless)
```

### Transfer to macOS:
```bash
# On Linux - compress the models
tar -czf qwen_gguf_models.tar.gz qwen_classifier_gguf/

# Transfer via SCP
scp qwen_gguf_models.tar.gz user@mac-hostname:~/Downloads/

# On macOS - extract
cd ~/Downloads
tar -xzf qwen_gguf_models.tar.gz
```

## Step 2: Import Model into LM Studio

### Method 1: Drag and Drop (Easiest)

1. **Open LM Studio**

2. **Navigate to the "My Models" tab**
   - Click on the folder icon on the left sidebar

3. **Drag and drop your GGUF file**
   - Drag `qwen-classifier-q4_k_m.gguf` into the LM Studio window
   - LM Studio will copy it to its model directory

4. **Wait for import to complete**
   - You'll see a progress indicator
   - Takes ~10-30 seconds depending on model size

### Method 2: Import Button

1. **Open LM Studio**

2. **Click "Import Model" or the "+" button**
   - Located in the "My Models" section

3. **Browse to your GGUF file**
   - Navigate to `qwen_classifier_gguf/`
   - Select `qwen-classifier-q4_k_m.gguf`

4. **Click "Open"**
   - LM Studio will import the model

### Method 3: Copy to LM Studio Directory (Advanced)

1. **Find LM Studio models directory:**
   ```bash
   # Default location on macOS
   ~/. lmstudio/models/

   # Create a folder for your model
   mkdir -p ~/.lmstudio/models/qwen-classifier
   ```

2. **Copy your GGUF file:**
   ```bash
   cp qwen_classifier_gguf/qwen-classifier-q4_k_m.gguf \
      ~/.lmstudio/models/qwen-classifier/
   ```

3. **Restart LM Studio**
   - Your model will appear in "My Models"

## Step 3: Load and Use Your Model

### Loading the Model

1. **Go to "Chat" tab** in LM Studio

2. **Click the model selector** (top dropdown)

3. **Find your model**
   - Look for "qwen-classifier-q4_k_m"
   - It may be under "Local Models" or "My Models"

4. **Click to load**
   - First load takes 5-10 seconds
   - You'll see loading progress

5. **Model is ready when you see:**
   - Green indicator
   - "Model loaded successfully"

### Using Your Classifier

#### Example 1: Simple Classification

**Prompt:**
```
Classify the following question into one of these categories: HR, IT, Sales, Finance, Operations

Question: How do I reset my password?
Category:
```

**Expected Response:**
```
IT
```

#### Example 2: Multiple Questions

**Prompt:**
```
Classify each question:

1. "How many vacation days do I have left?"
2. "I can't log into my email"
3. "What's our Q4 revenue?"
4. "I need a quote for a customer"

Categories: HR, IT, Sales, Finance, Operations
```

#### Example 3: Detailed Classification

**Prompt:**
```
Classify this question and explain why:

Question: "My laptop won't connect to the VPN"

Categories: HR, IT, Sales, Finance, Operations, Legal, Marketing, Customer Support
```

## Step 4: Advanced Settings

### Recommended Settings for Classification

1. **Click "Settings" icon** (⚙️) in chat interface

2. **Adjust parameters:**

   **For consistent classifications:**
   - **Temperature:** `0.1` (more deterministic)
   - **Top P:** `0.9`
   - **Max Tokens:** `10` (only need category name)
   - **Stop Sequences:** Add `\n` (stop after category)

   **For varied/creative responses:**
   - **Temperature:** `0.7` (more variety)
   - **Top P:** `0.95`
   - **Max Tokens:** `50-100`

3. **Context Length:**
   - **Default:** `2048` (sufficient for classification)
   - Increase if classifying very long texts

### Prompt Templates

Create reusable templates in LM Studio:

#### Template 1: Basic Classification
```
Classify this question into one of these categories: {CATEGORIES}

Question: {QUESTION}
Category:
```

#### Template 2: Classification with Confidence
```
Classify this question and rate your confidence (0-100%):

Categories: {CATEGORIES}
Question: {QUESTION}

Format: Category (Confidence%)
```

#### Template 3: Batch Classification
```
Classify each question. Respond with only the category for each.

Categories: {CATEGORIES}

{QUESTION_LIST}
```

## Step 5: Using LM Studio Server (API Mode)

Turn your local model into an API server!

### Enable Server Mode

1. **Go to "Developer" or "Server" tab** in LM Studio

2. **Click "Start Server"**
   - Default port: `1234`
   - Endpoint: `http://localhost:1234/v1`

3. **Your API is now running!**

### Test with cURL

```bash
curl http://localhost:1234/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {
        "role": "user",
        "content": "Classify: How do I reset my password?\nCategories: HR, IT, Sales"
      }
    ],
    "temperature": 0.1,
    "max_tokens": 10
  }'
```

### Python Client Example

```python
import requests

def classify_question(question, categories):
    """Classify a question using local LM Studio API."""

    prompt = f"""Classify the following question into one of these categories: {', '.join(categories)}

Question: {question}
Category:"""

    response = requests.post(
        "http://localhost:1234/v1/chat/completions",
        json={
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.1,
            "max_tokens": 10,
        }
    )

    result = response.json()
    category = result["choices"][0]["message"]["content"].strip()
    return category

# Example usage
categories = ["HR", "IT", "Sales", "Finance"]
question = "How many vacation days do I have?"

result = classify_question(question, categories)
print(f"Category: {result}")
```

### JavaScript/Node.js Client Example

```javascript
async function classifyQuestion(question, categories) {
  const prompt = `Classify the following question into one of these categories: ${categories.join(', ')}

Question: ${question}
Category:`;

  const response = await fetch('http://localhost:1234/v1/chat/completions', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      messages: [{ role: 'user', content: prompt }],
      temperature: 0.1,
      max_tokens: 10,
    }),
  });

  const data = await response.json();
  return data.choices[0].message.content.trim();
}

// Example usage
const categories = ['HR', 'IT', 'Sales', 'Finance'];
const question = 'How many vacation days do I have?';

classifyQuestion(question, categories)
  .then(category => console.log(`Category: ${category}`));
```

## Step 6: Model Management

### Check Model Performance

1. **Monitor in LM Studio:**
   - **Speed:** Tokens per second (should be 50-100 for Q4_K_M on M1)
   - **Memory:** RAM usage (will show in Activity Monitor)
   - **Temperature:** CPU temperature

2. **Activity Monitor (macOS):**
   ```
   Open Activity Monitor → Search "LM Studio"
   - Memory: Should be < 2GB for Q4_K_M model
   - CPU: Will spike during generation, then drop
   ```

### Switch Between Quantizations

If one quantization doesn't work well:

| Quantization | Size | Speed | Quality | Best For |
|--------------|------|-------|---------|----------|
| Q4_K_M | 150-200MB | Fast (50-100 tok/s) | Good | **Recommended** - Best balance |
| Q5_K_M | 200-250MB | Medium (40-80 tok/s) | Better | Higher accuracy needed |
| Q8_0 | 300-350MB | Slower (30-50 tok/s) | Best | Maximum accuracy |
| F16 | 500MB | Slowest (20-40 tok/s) | Perfect | Benchmarking |

### Unload/Remove Models

**To unload (free memory):**
- Click model selector → "Unload Model"

**To remove completely:**
1. Go to "My Models" tab
2. Right-click on your model
3. Select "Delete" or "Remove"

## Troubleshooting

### Model Won't Load

**Error: "Failed to load model"**

**Solution 1:** Check GGUF file integrity
```bash
# Check file size
ls -lh qwen-classifier-q4_k_m.gguf

# Should be 150-200MB for Q4_K_M
# If much smaller, re-export the model
```

**Solution 2:** Try different quantization
- If Q4_K_M fails, try Q5_K_M or F16
- Some quantizations may be incompatible

**Solution 3:** Update LM Studio
- Download latest version from https://lmstudio.ai
- Older versions may not support newer GGUF formats

### Slow Performance

**Issue: < 20 tokens/sec**

**Solutions:**
1. **Use lighter quantization:** Q4_K_M instead of Q8_0
2. **Close other apps:** Free up RAM
3. **Check Activity Monitor:** Ensure no background processes
4. **Restart LM Studio:** Clears memory leaks

### Wrong Classifications

**Model gives incorrect categories**

**Solutions:**
1. **Adjust temperature:** Lower to 0.1 for more consistency
2. **Improve prompt:** Be more specific about categories
3. **Check training data:** May need to retrain with more examples
4. **Try higher quality quantization:** Q5_K_M or Q8_0

### Can't Find Model in List

**Model doesn't appear in dropdown**

**Solutions:**
1. **Restart LM Studio:** Refresh model cache
2. **Check import location:** Verify file copied to `~/.lmstudio/models/`
3. **Check file extension:** Must be `.gguf`
4. **Re-import:** Try drag-and-drop again

## Tips and Best Practices

### 1. Organize Your Models

Create folders in LM Studio models directory:
```bash
~/.lmstudio/models/
├── qwen-classifier/
│   ├── qwen-classifier-q4_k_m.gguf
│   ├── qwen-classifier-q5_k_m.gguf
│   └── README.txt  # Notes about this model
├── other-models/
└── ...
```

### 2. Document Your Prompts

Keep a text file with working prompts:
```
prompts.txt:
- Classification prompt: "Classify this question into..."
- Confidence prompt: "Classify and rate confidence..."
- Batch prompt: "Classify each of these questions..."
```

### 3. Test Different Settings

Experiment to find optimal settings:
- Start with temperature 0.1
- Increase if responses too similar
- Decrease if responses too varied

### 4. Benchmark Performance

Test your model with known examples:
```bash
# Create test_cases.txt with known Q&A pairs
# Run through LM Studio API
# Calculate accuracy
```

### 5. Version Control

Keep track of different model versions:
```
qwen-classifier-v1-q4_k_m.gguf  # First training
qwen-classifier-v2-q4_k_m.gguf  # More data
qwen-classifier-v3-q4_k_m.gguf  # Fine-tuned
```

## Next Steps

1. **Test your model** with example questions
2. **Adjust prompts** for better results
3. **Enable API mode** for application integration
4. **Monitor performance** and optimize settings
5. **Retrain if needed** with more/better data

## Resources

- **LM Studio Docs:** https://lmstudio.ai/docs
- **GGUF Format:** https://github.com/ggerganov/ggml/blob/master/docs/gguf.md
- **llama.cpp:** https://github.com/ggerganov/llama.cpp
- **Your Training Pipeline:** See [README.md](README.md)

## Support

For issues with:
- **Model export:** See [export_to_gguf.py](export_to_gguf.py)
- **Training:** See [README.md](README.md) and [LINUX_SETUP.md](LINUX_SETUP.md)
- **LM Studio:** Visit https://lmstudio.ai/support

---

**Happy Classifying! 🚀**
