import { readFileSync } from 'node:fs';
import * as path from 'node:path';

import { imageSize } from 'image-size';
import { createHighlighter } from 'shiki';
import { toString } from 'hast-util-to-string';
import rehypeKatex from 'rehype-katex';
import rehypeRaw from 'rehype-raw';
import rehypeSlug from 'rehype-slug';
import rehypeShikiFromHighlighter from '@shikijs/rehype/core';
import remarkGfm from 'remark-gfm';
import remarkMath from 'remark-math';
import remarkParse from 'remark-parse';
import remarkRehype from 'remark-rehype';
import remarkSmartypants from 'remark-smartypants';
import { unified } from 'unified';
import { visit } from 'unist-util-visit';
import type { Element, ElementContent, Parent, Root } from 'hast';
import type { VFile } from 'vfile';
import rehypeStringify from 'rehype-stringify';

import { toReference, type BibIndex } from './bibtex';
import type { Reference, TocItem } from '$lib/types';

type MarkdownData = {
	toc?: TocItem[];
	references?: Reference[];
	bib?: BibIndex;
};

type MarkdownProcessor = Awaited<ReturnType<typeof createProcessor>>;

let processorPromise: Promise<MarkdownProcessor> | undefined;

function getProperty(node: Element, name: string): string | undefined {
	const value = node.properties?.[name];
	if (typeof value === 'string') {
		return value;
	}
	if (Array.isArray(value)) {
		return value.join(',');
	}
	return value === undefined || value === null ? undefined : String(value);
}

function textNode(value: string): ElementContent {
	return { type: 'text', value };
}

function citationAnchor(key: string, index: number): ElementContent {
	return {
		type: 'element',
		tagName: 'a',
		properties: {
			className: ['citation'],
			href: `#ref-${key}`
		},
		children: [textNode(`[${index}]`)]
	};
}

function unknownCitation(key: string): ElementContent {
	console.warn(`Unknown bibliography key "${key}"`);
	return textNode('[?]');
}

function replaceCitations(
	parent: Parent,
	bib: BibIndex | undefined,
	references: Reference[],
	numbers: Map<string, number>
): void {
	const children: ElementContent[] = [];

	for (const child of parent.children as ElementContent[]) {
		if (child.type === 'element' && child.tagName === 'd-cite') {
			const keys = (getProperty(child, 'key') ?? '')
				.split(',')
				.map((key) => key.trim())
				.filter(Boolean);

			keys.forEach((key, keyIndex) => {
				if (keyIndex > 0) {
					children.push(textNode(', '));
				}

				const existing = numbers.get(key);
				if (existing !== undefined) {
					children.push(citationAnchor(key, existing));
					return;
				}

				const entry = bib?.[key];
				if (!entry) {
					children.push(unknownCitation(key));
					return;
				}

				const index = references.length + 1;
				numbers.set(key, index);
				const reference = toReference(entry);
				references.push({ key, index, html: reference.html, url: reference.url });
				children.push(citationAnchor(key, index));
			});
			continue;
		}

		if (child.type === 'element') {
			replaceCitations(child, bib, references, numbers);
		}
		children.push(child);
	}

	parent.children = children as typeof parent.children;
}

function citationAndTocPlugin() {
	return (tree: Root, file: VFile): void => {
		const data = file.data as MarkdownData;
		const toc: TocItem[] = [];
		visit(tree, 'element', (node: Element) => {
			if ((node.tagName === 'h2' || node.tagName === 'h3') && getProperty(node, 'id')) {
				toc.push({
					id: getProperty(node, 'id') ?? '',
					title: toString(node),
					depth: node.tagName === 'h2' ? 2 : 3
				});
			}
		});

		const references: Reference[] = [];
		replaceCitations(tree, data.bib, references, new Map<string, number>());
		data.toc = toc;
		data.references = references;
	};
}

/**
 * Stamps intrinsic dimensions onto local images so the browser can reserve space
 * before they load; without this, lazy figures shift the article and break
 * in-page anchors.
 */
function imageDimensionsPlugin() {
	return (tree: Root): void => {
		visit(tree, 'element', (node: Element) => {
			if (node.tagName !== 'img') {
				return;
			}
			const source = getProperty(node, 'src');
			if (!source?.startsWith('/assets/')) {
				return;
			}
			if (node.properties?.width !== undefined || node.properties?.height !== undefined) {
				return;
			}
			try {
				const file = path.join(process.cwd(), 'static', source);
				const { width, height } = imageSize(readFileSync(file));
				if (width && height) {
					node.properties = { ...node.properties, width, height };
				}
			} catch (error) {
				console.warn(`Could not read image dimensions for ${source}: ${String(error)}`);
			}
		});
	};
}

function externalLinkPlugin() {
	return (tree: Root): void => {
		visit(tree, 'element', (node: Element) => {
			if (node.tagName !== 'a') {
				return;
			}
			const href = getProperty(node, 'href');
			if (href?.startsWith('http')) {
				node.properties = {
					...node.properties,
					target: '_blank',
					rel: ['external', 'nofollow', 'noopener']
				};
			}
		});
	};
}

async function createProcessor() {
	const highlighter = await createHighlighter({
		themes: ['github-light', 'github-dark-dimmed'],
		langs: ['bash', 'css', 'html', 'javascript', 'json', 'markdown', 'python', 'typescript', 'yaml']
	});

	return unified()
		.use(remarkParse)
		.use(remarkGfm)
		.use(remarkMath)
		.use(remarkSmartypants)
		.use(remarkRehype, { allowDangerousHtml: true })
		.use(rehypeRaw)
		.use(imageDimensionsPlugin)
		.use(rehypeSlug)
		.use(rehypeKatex)
		.use(rehypeShikiFromHighlighter, highlighter, {
			themes: {
				light: 'github-light',
				dark: 'github-dark-dimmed'
			},
			fallbackLanguage: 'text'
		})
		.use(citationAndTocPlugin)
		.use(externalLinkPlugin)
		.use(rehypeStringify, { allowDangerousHtml: true });
}

async function getProcessor(): Promise<MarkdownProcessor> {
	processorPromise ??= createProcessor();
	return processorPromise;
}

export async function renderMarkdown(
	source: string,
	opts: { bib?: BibIndex } = {}
): Promise<{ html: string; toc: TocItem[]; references: Reference[] }> {
	const processor = await getProcessor();
	const file = await processor.process({
		value: source,
		data: { bib: opts.bib } satisfies MarkdownData
	});
	const data = file.data as MarkdownData;
	return {
		html: String(file),
		toc: data.toc ?? [],
		references: data.references ?? []
	};
}
