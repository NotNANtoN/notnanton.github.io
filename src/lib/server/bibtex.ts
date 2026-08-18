import { readFileSync } from 'node:fs';
import * as path from 'node:path';

import type { Publication, PublicationLink } from '$lib/types';

export type BibEntry = {
	key: string;
	type: string;
	fields: Record<string, string>;
	raw: string;
};

export type BibIndex = Record<string, BibEntry>;

const INTERNAL_FIELDS = new Set([
	'abbr',
	'bibtex_show',
	'preview',
	'selected',
	'html',
	'website',
	'pdf',
	'code',
	'slides',
	'video',
	'poster',
	'arxiv',
	'doi'
]);

const ACCENTS: Record<string, Record<string, string>> = {
	'"': { a: 'ä', e: 'ë', i: 'ï', o: 'ö', u: 'ü', A: 'Ä', E: 'Ë', I: 'Ï', O: 'Ö', U: 'Ü' },
	"'": { a: 'á', e: 'é', i: 'í', o: 'ó', u: 'ú', A: 'Á', E: 'É', I: 'Í', O: 'Ó', U: 'Ú' },
	'`': { a: 'à', e: 'è', i: 'ì', o: 'ò', u: 'ù', A: 'À', E: 'È', I: 'Ì', O: 'Ò', U: 'Ù' },
	'~': { a: 'ã', n: 'ñ', o: 'õ', A: 'Ã', N: 'Ñ', O: 'Õ' },
	'^': { a: 'â', e: 'ê', i: 'î', o: 'ô', u: 'û', A: 'Â', E: 'Ê', I: 'Î', O: 'Ô', U: 'Û' },
	'=': { a: 'ā', e: 'ē', i: 'ī', o: 'ō', u: 'ū', A: 'Ā', E: 'Ē', I: 'Ī', O: 'Ō', U: 'Ū' },
	'.': { z: 'ż', Z: 'Ż' },
	'v': { c: 'č', s: 'š', z: 'ž', C: 'Č', S: 'Š', Z: 'Ž' }
};

function findEntryEnd(source: string, openingBrace: number): number {
	let depth = 0;
	let quoted = false;

	for (let index = openingBrace; index < source.length; index += 1) {
		const character = source[index];
		// A backslash escapes the next character, so LaTeX such as R{\"o}der does not
		// look like the start of a quoted value.
		if (character === '\\') {
			index += 1;
			continue;
		}

		if (quoted) {
			if (character === '"') {
				quoted = false;
			}
			continue;
		}

		// Quoted values only occur at field level; deeper braces contain literal quotes.
		if (character === '"' && depth === 1) {
			quoted = true;
		} else if (character === '{') {
			depth += 1;
		} else if (character === '}') {
			depth -= 1;
			if (depth === 0) {
				return index;
			}
		}
	}

	return -1;
}

function rawEntries(source: string): string[] {
	const entries: string[] = [];
	const pattern = /@([a-zA-Z]+)\s*\{\s*([^,\s]+)\s*,/g;
	let match: RegExpExecArray | null;

	while ((match = pattern.exec(source)) !== null) {
		const openingBrace = source.indexOf('{', match.index);
		const end = findEntryEnd(source, openingBrace);
		if (openingBrace < 0 || end < 0) {
			throw new Error(`Malformed BibTeX entry near offset ${match.index}`);
		}
		entries.push(source.slice(match.index, end + 1));
		pattern.lastIndex = end + 1;
	}

	return entries;
}

function stripOuterDelimiters(value: string): string {
	let result = value.trim();
	let changed = true;
	while (changed && result.length >= 2) {
		changed = false;
		if (
			(result.startsWith('{') && result.endsWith('}')) ||
			(result.startsWith('"') && result.endsWith('"'))
		) {
			result = result.slice(1, -1).trim();
			changed = true;
		}
	}
	return result;
}

function decodeLatex(value: string): string {
	let result = stripOuterDelimiters(value);
	result = result.replace(
		/\\(["'`~^=.v])\s*\{?([A-Za-z])\}?/g,
		(_match, accent: string, letter: string) => ACCENTS[accent]?.[letter] ?? letter
	);
	result = result
		.replace(/\\aa\b/g, 'å')
		.replace(/\\AA\b/g, 'Å')
		.replace(/\\ae\b/g, 'æ')
		.replace(/\\AE\b/g, 'Æ')
		.replace(/\\oe\b/g, 'œ')
		.replace(/\\OE\b/g, 'Œ')
		.replace(/\\o\b/g, 'ø')
		.replace(/\\O\b/g, 'Ø')
		.replace(/\\ss\b/g, 'ß')
		.replace(/\\&/g, '&')
		.replace(/\\%/g, '%')
		.replace(/\\_/g, '_')
		.replace(/\\#/g, '#')
		.replace(/\\textasciitilde/g, '~')
		.replace(/\\textbackslash/g, '\\')
		.replace(/---/g, '—')
		.replace(/(^|[^-])--([^-]|$)/g, '$1–$2')
		.replace(/[{}]/g, '');
	return result.replace(/\s+/g, ' ').trim();
}

function readDelimitedValue(source: string, start: number): { value: string; next: number } {
	const first = source[start];
	if (first === '{') {
		let depth = 0;
		for (let index = start; index < source.length; index += 1) {
			if (source[index] === '\\') {
				index += 1;
			} else if (source[index] === '{') {
				depth += 1;
			} else if (source[index] === '}') {
				depth -= 1;
				if (depth === 0) {
					return { value: source.slice(start, index + 1), next: index + 1 };
				}
			}
		}
		throw new Error('Unclosed braced BibTeX value');
	}

	if (first === '"') {
		let escaped = false;
		for (let index = start + 1; index < source.length; index += 1) {
			const character = source[index];
			if (escaped) {
				escaped = false;
			} else if (character === '\\') {
				escaped = true;
			} else if (character === '"') {
				return { value: source.slice(start, index + 1), next: index + 1 };
			}
		}
		throw new Error('Unclosed quoted BibTeX value');
	}

	let end = start;
	while (
		end < source.length &&
		source[end] !== ',' &&
		source[end] !== '\n' &&
		source[end] !== '\r'
	) {
		end += 1;
	}
	return { value: source.slice(start, end), next: end };
}

function parseFields(raw: string): { key: string; type: string; fields: Record<string, string> } {
	const header = /^@([a-zA-Z]+)\s*\{\s*([^,\s]+)\s*,/.exec(raw);
	if (!header) {
		throw new Error('Malformed BibTeX entry header');
	}

	const fields: Record<string, string> = {};
	let cursor = header[0].length;
	while (cursor < raw.length - 1) {
		while (cursor < raw.length - 1 && /[\s,]/.test(raw[cursor])) {
			cursor += 1;
		}
		if (raw[cursor] === '}') {
			break;
		}

		const fieldStart = cursor;
		while (cursor < raw.length && /[A-Za-z0-9_:-]/.test(raw[cursor])) {
			cursor += 1;
		}
		if (fieldStart === cursor) {
			throw new Error(`Malformed BibTeX field near "${raw.slice(cursor, cursor + 20)}"`);
		}
		const field = raw.slice(fieldStart, cursor).toLowerCase();
		while (cursor < raw.length && /\s/.test(raw[cursor])) {
			cursor += 1;
		}
		if (raw[cursor] !== '=') {
			throw new Error(`Missing equals sign for BibTeX field "${field}"`);
		}
		cursor += 1;
		while (cursor < raw.length && /\s/.test(raw[cursor])) {
			cursor += 1;
		}
		const parsed = readDelimitedValue(raw, cursor);
		fields[field] = decodeLatex(parsed.value);
		cursor = parsed.next;
	}

	return { key: header[2], type: header[1].toLowerCase(), fields };
}

function field(entry: BibEntry, name: string): string | undefined {
	return entry.fields[name.toLowerCase()];
}

function escapeHtml(value: string): string {
	return value
		.replace(/&/g, '&amp;')
		.replace(/</g, '&lt;')
		.replace(/>/g, '&gt;')
		.replace(/"/g, '&quot;')
		.replace(/'/g, '&#39;');
}

function splitAuthors(value: string): string[] {
	return value
		.split(/\s+and\s+/i)
		.map((author) => {
			const normalized = decodeLatex(author);
			const comma = normalized.indexOf(',');
			if (comma < 0) {
				return normalized;
			}
			const last = normalized.slice(0, comma).trim();
			const first = normalized.slice(comma + 1).trim();
			return `${first} ${last}`.trim();
		})
		.filter(Boolean);
}

function boolField(value: string | undefined): boolean {
	return value?.toLowerCase() === 'true' || value === '1';
}

function normalizeExternalUrl(value: string, prefix?: string): string {
	if (/^https?:\/\//i.test(value)) {
		return value;
	}
	return `${prefix ?? ''}${value}`;
}

function publicationLinks(entry: BibEntry): PublicationLink[] {
	const links: PublicationLink[] = [];
	const arxiv = field(entry, 'arxiv');
	if (arxiv) {
		links.push({ name: 'arXiv', url: normalizeExternalUrl(arxiv, 'https://arxiv.org/abs/') });
	}

	const doi = field(entry, 'doi');
	if (doi) {
		links.push({ name: 'DOI', url: normalizeExternalUrl(doi, 'https://doi.org/') });
	}

	const namedFields: Array<[string, string]> = [
		['html', 'Website'],
		['website', 'Website'],
		['pdf', 'PDF'],
		['code', 'Code'],
		['slides', 'Slides'],
		['video', 'Video'],
		['poster', 'Poster']
	];
	for (const [fieldName, label] of namedFields) {
		const value = field(entry, fieldName);
		if (value) {
			links.push({ name: label, url: value });
		}
	}
	return links;
}

function venueHtml(entry: BibEntry): string {
	const journal = field(entry, 'journal');
	const booktitle = field(entry, 'booktitle');
	const school = field(entry, 'school');
	const volume = field(entry, 'volume');
	const number = field(entry, 'number');
	const pages = field(entry, 'pages');

	if (entry.type === 'article' && journal) {
		let result = `<em>${escapeHtml(journal)}</em>`;
		if (volume) {
			result += `, ${escapeHtml(volume)}`;
			if (number) {
				result += `(${escapeHtml(number)})`;
			}
		} else if (number) {
			result += `, (${escapeHtml(number)})`;
		}
		if (pages) {
			result += `, ${escapeHtml(pages)}`;
		}
		return result;
	}

	if (entry.type === 'inproceedings' && booktitle) {
		return `<em>${escapeHtml(booktitle)}</em>`;
	}

	if (entry.type === 'mastersthesis' || entry.type === 'phdthesis') {
		const thesisType =
			field(entry, 'type') ??
			(entry.type === 'phdthesis' ? 'PhD Thesis' : "Master's Thesis");
		return `${school ? `<em>${escapeHtml(school)}</em>` : ''}${school ? ', ' : ''}${escapeHtml(
			thesisType
		)}`;
	}

	const fallback = journal ?? booktitle ?? school ?? field(entry, 'publisher') ?? '';
	return fallback ? `<em>${escapeHtml(fallback)}</em>` : '';
}

function entryYear(entry: BibEntry): number {
	const value = Number.parseInt(field(entry, 'year') ?? '0', 10);
	return Number.isFinite(value) ? value : 0;
}

export function loadBib(file: string): BibIndex {
	const resolved = path.isAbsolute(file) ? file : path.resolve(file);
	let source: string;
	try {
		source = readFileSync(resolved, 'utf8');
	} catch (error) {
		throw new Error(`Unable to read bibliography ${resolved}: ${String(error)}`);
	}

	try {
		const index: BibIndex = {};
		for (const raw of rawEntries(source)) {
			const parsedEntry = parseFields(raw);
			index[parsedEntry.key] = { ...parsedEntry, raw: raw.trim() };
		}
		return index;
	} catch (error) {
		throw new Error(`Unable to parse bibliography ${resolved}: ${String(error)}`);
	}
}

export function toPublications(index: BibIndex): Publication[] {
	return Object.values(index)
		.map((entry): Publication => ({
			key: entry.key,
			type: entry.type,
			title: field(entry, 'title') ?? entry.key,
			authors: splitAuthors(field(entry, 'author') ?? ''),
			venueHtml: venueHtml(entry),
			year: entryYear(entry),
			abbr: field(entry, 'abbr'),
			selected: boolField(field(entry, 'selected')),
			bibtex: entry.raw,
			links: publicationLinks(entry)
		}))
		.sort((left, right) => right.year - left.year || left.title.localeCompare(right.title));
}

export function toReference(entry: BibEntry): { html: string; url?: string } {
	const authors = splitAuthors(field(entry, 'author') ?? '');
	const authorText = authors.length > 0 ? authors.join(', ') : 'Unknown authors';
	const year = field(entry, 'year') ?? 'n.d.';
	const title = escapeHtml(field(entry, 'title') ?? entry.key);
	const url = field(entry, 'arxiv')
		? normalizeExternalUrl(field(entry, 'arxiv') ?? '', 'https://arxiv.org/abs/')
		: field(entry, 'doi')
			? normalizeExternalUrl(field(entry, 'doi') ?? '', 'https://doi.org/')
			: undefined;
	return {
		html: `${escapeHtml(authorText)} (${escapeHtml(year)}). ${title}. ${venueHtml(entry)}.`,
		url
	};
}

export function publicFields(entry: BibEntry): Record<string, string> {
	return Object.fromEntries(
		Object.entries(entry.fields).filter(([name]) => !INTERNAL_FIELDS.has(name))
	);
}
