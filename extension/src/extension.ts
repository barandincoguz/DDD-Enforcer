import * as vscode from "vscode";
import axios from "axios";

// Backend'den gelecek veri yapısı
interface Violation {
  type: string;
  message: string;
  suggestion: string;
}

interface ValidationResponse {
  is_violation: boolean;
  violations: Violation[];
}

export function activate(context: vscode.ExtensionContext) {
  console.log("DDD Enforcer is now active!");

  // Hataları göstereceğimiz koleksiyon (Diagnostic Collection)
  const diagnosticCollection =
    vscode.languages.createDiagnosticCollection("ddd-enforcer");
  context.subscriptions.push(diagnosticCollection);

  // Kaydetme olayını dinle (CTRL+S basınca çalışır)
  context.subscriptions.push(
    vscode.workspace.onDidSaveTextDocument(async (document) => {
      if (document.languageId !== "python") {
        return; // Sadece Python dosyalarına bak
      }
      await validateCode(document, diagnosticCollection);
    })
  );
}

async function validateCode(
  document: vscode.TextDocument,
  collection: vscode.DiagnosticCollection
) {
  // Önceki hataları temizle
  collection.clear();

  const codeContent = document.getText();
  const fileName = document.fileName;

  try {
    // Backend'e istek at (Senin backend portun 8000)
    const response = await axios.post<ValidationResponse>(
      "http://127.0.0.1:8000/validate",
      {
        filename: fileName,
        content: codeContent,
      }
    );

    const data = response.data;

    if (data.is_violation && data.violations) {
      const diagnostics: vscode.Diagnostic[] = [];

      data.violations.forEach((violation) => {
        // Hatanın nerede olduğunu bulmaya çalışalım.
        // Basitlik için dosyanın ilk satırını işaretliyoruz.
        // İleri seviye versiyonda AST'den satır numarası da dönebiliriz.

        // Mesajın içinde geçen kelimeyi (örn: Client) bulup onu çizelim
        const keyword = extractKeyword(violation.message);
        const range = findKeywordRange(document, keyword);

        const diagnostic = new vscode.Diagnostic(
          range,
          `[DDD Violation]: ${violation.message} \n💡 Suggestion: ${violation.suggestion}`,
          vscode.DiagnosticSeverity.Error
        );

        diagnostic.source = "DDD Enforcer";
        diagnostics.push(diagnostic);
      });

      collection.set(document.uri, diagnostics);
    }
  } catch (error) {
    console.error("Error validating code:", error);
    vscode.window.showErrorMessage(
      "DDD Enforcer: Could not connect to backend server."
    );
  }
}

// Hata mesajından anahtar kelimeyi tahmin et (Basit regex)
function extractKeyword(message: string): string {
  // "Class name 'ClientManager' uses..." -> ClientManager'ı yakala
  const match = message.match(/'([^']+)'/);
  return match ? match[1] : "";
}

// Dosya içinde kelimenin geçtiği ilk yeri bul
function findKeywordRange(
  document: vscode.TextDocument,
  keyword: string
): vscode.Range {
  if (!keyword) {
    return new vscode.Range(0, 0, 0, 0); // Bulamazsa ilk satır
  }

  const text = document.getText();
  const index = text.indexOf(keyword);

  if (index === -1) {
    return new vscode.Range(0, 0, 0, 0);
  }

  const positionStart = document.positionAt(index);
  const positionEnd = document.positionAt(index + keyword.length);
  return new vscode.Range(positionStart, positionEnd);
}

export function deactivate() {}
