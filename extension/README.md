# DDD Enforcer

**Validate your Python code against Domain-Driven Design rules extracted from your SRS documents.**

DDD Enforcer is a VS Code extension that uses AI to analyze your Software Requirements Specification (SRS) documents and automatically enforces Domain-Driven Design principles in your codebase.

## ✨ Features

- 🤖 **AI-Powered Domain Model Generation**: Automatically extract bounded contexts, entities, value objects, and domain rules from your SRS/design documents
- ✅ **Real-time Validation**: Validates your Python code on every save
- 📚 **Source References**: See exactly which part of your SRS document each violation relates to
- 🎯 **Smart Diagnostics**: Shows violations with suggestions and quick fixes
- 🔍 **RAG-Powered**: Uses Retrieval-Augmented Generation for accurate source tracking

## 📋 Prerequisites

Before using DDD Enforcer, ensure you have:

1. **Python 3.10+** installed on your system
2. **Gemini API Key** from [Google AI Studio](https://aistudio.google.com/)

## 🚀 Installation

### Step 1: Install the Extension

Install DDD Enforcer from the VS Code Marketplace or install manually:

```bash
code --install-extension ddd-enforcer-1.0.0.vsix
```

### Step 2: Install Python Dependencies

Open a terminal and navigate to the extension's backend folder:

```bash
# Find the extension path (usually in ~/.vscode/extensions/ddd-enforcer-x.x.x/)
cd ~/.vscode/extensions/ddd-enforcer-*/backend

# Install dependencies
pip install -r requirements.txt
```

### Step 3: Configure API Key

You have three options to provide your Gemini API Key:

1. **VS Code Settings** (Recommended):
   - Open Settings (`Cmd+,` / `Ctrl+,`)
   - Search for "DDD Enforcer"
   - Enter your API key in the "Gemini Api Key" field

2. **Environment Variable**:

   ```bash
   export GEMINI_API_KEY="your-api-key-here"
   ```

3. **Prompted on First Use**: The extension will ask for your API key when needed

## 📖 Usage

### 1. Initialize Domain Model

Before validation can work, you need to generate a domain model from your SRS documents:

1. Open Command Palette (`Cmd+Shift+P` / `Ctrl+Shift+P`)
2. Run: **DDD Enforcer: Initialize Domain Model**
3. Select one or more SRS/design documents (PDF, DOCX, or TXT)
4. Wait for the AI to analyze and generate the domain model

The domain model will be saved to `./domain/model.json` in your workspace.

### 2. Validate Code

Once the domain model is generated:

- **Automatic**: Simply save any Python file (`Cmd+S` / `Ctrl+S`) - validation runs automatically
- **Manual**: Run **DDD Enforcer: Validate Current File** from Command Palette

### 3. View Violations

Violations appear as red underlines in your code. Hover to see:

- The violation type and message
- Suggested fix
- Source references from your SRS document

Click on source references (Quick Fix menu) to jump directly to the relevant section in your SRS document.

## ⚙️ Settings

| Setting                       | Description                       | Default    |
| ----------------------------- | --------------------------------- | ---------- |
| `ddd-enforcer.geminiApiKey`   | Your Gemini API Key               | `""`       |
| `ddd-enforcer.backendPort`    | Preferred port for backend server | `8000`     |
| `ddd-enforcer.pythonPath`     | Path to Python executable         | `"python"` |
| `ddd-enforcer.validateOnSave` | Auto-validate on file save        | `true`     |
| `ddd-enforcer.showStatusBar`  | Show status in status bar         | `true`     |

## 🎯 Commands

| Command                                 | Description                               |
| --------------------------------------- | ----------------------------------------- |
| `DDD Enforcer: Initialize Domain Model` | Generate domain model from SRS documents  |
| `DDD Enforcer: Validate Current File`   | Manually validate the current Python file |
| `DDD Enforcer: Show Status`             | Show backend and domain model status      |
| `DDD Enforcer: Restart Backend Server`  | Restart the backend server                |

## 🔧 Troubleshooting

### Backend won't start

1. Check Python is installed: `python --version`
2. Ensure dependencies are installed: `pip install -r requirements.txt`
3. Check the Output panel (`View > Output > DDD Enforcer`) for error logs

### No violations showing

1. Ensure domain model exists (`./domain/model.json`)
2. Check the backend is running (status bar shows ✓)
3. Try restarting the backend: **DDD Enforcer: Restart Backend Server**

### API Key issues

- Ensure your API key is valid and has access to Gemini API
- Try setting it via environment variable for debugging

## 📄 Supported Document Types

- PDF (`.pdf`)
- Microsoft Word (`.docx`)
- Plain Text (`.txt`)

## 🗺️ Roadmap

- [ ] Multi-language support (Java, TypeScript, C#)
- [ ] More document formats (Markdown, JSON)
- [ ] Custom rule definitions
- [ ] Team/enterprise features

## 📝 License

MIT License - see LICENSE file for details.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit issues and pull requests.

---

**Made with ❤️ for Domain-Driven Design enthusiasts**
