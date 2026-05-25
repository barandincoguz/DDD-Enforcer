import { defineConfig } from '@vscode/test-cli';
import * as path from 'path';
import { fileURLToPath } from 'url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));

export default defineConfig({
	files: 'out/test/**/*.test.js',
	launchArgs: [
		'--user-data-dir', path.resolve(__dirname, 'out/t'),
		'--extensions-dir', path.resolve(__dirname, 'out/e'),
	],
});
