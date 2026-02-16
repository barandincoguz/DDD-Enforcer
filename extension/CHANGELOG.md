# Changelog

All notable changes to "DDD Enforcer" extension will be documented in this file.

## [1.0.0] - 2026-01-21

### Added

- 🎉 Initial release
- AI-powered domain model generation from SRS documents (PDF, DOCX, TXT)
- Real-time Python code validation on save
- RAG-powered source references for violations
- Status bar integration showing validation state
- Quick fix actions to view source documents
- Configurable settings (API key, port, Python path)
- Lazy-start backend for improved performance
- Auto port detection when default port is occupied

### Commands

- `DDD Enforcer: Initialize Domain Model` - Generate domain model from documents
- `DDD Enforcer: Validate Current File` - Manual validation trigger
- `DDD Enforcer: Show Status` - View backend and model status
- `DDD Enforcer: Restart Backend Server` - Restart the backend

## [Unreleased]

### Planned

- Multi-language support (Java, TypeScript, C#)
- More document formats (Markdown)
- Custom rule definitions
- Performance optimizations
