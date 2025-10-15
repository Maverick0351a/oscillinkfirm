# Local LLM models (GGUF)

This folder holds quantized CPU-friendly models for fully offline laptop demos.

Recommended defaults:

- Phi-3-mini-4k-instruct — quant: Q4_K_M — file: `phi3-mini.Q4_K_M.gguf`
- Optional: Mistral-7B-Instruct — quant: Q4_K_M — file: `mistral-7b-instruct.Q4_K_M.gguf`

Download hints:

- Hugging Face or other mirrors hosting GGUF exports. Example search terms:
  - "Phi-3-mini-4k-instruct GGUF Q4_K_M"
  - "Mistral-7B-Instruct GGUF Q4_K_M"

Verify SHA-256 after download:

```powershell
# Windows PowerShell
Get-FileHash .\phi3-mini.Q4_K_M.gguf -Algorithm SHA256 | Format-List
```

```bash
# macOS/Linux
shasum -a 256 ./phi3-mini.Q4_K_M.gguf
```

Place the GGUF files here, then run the CPU compose stack to serve an OpenAI-compatible API at http://127.0.0.1:8001/v1.
