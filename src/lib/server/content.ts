import { existsSync, readdirSync, readFileSync } from 'node:fs';
import * as path from 'node:path';

import matter from 'gray-matter';
import readingTime from 'reading-time';
import { parse as parseYaml } from 'yaml';

import type {
	About,
	AboutProfile,
	Author,
	CVEntry,
	CVSection,
	NewsItem,
	Post,
	PostMeta,
	Publication,
	Project,
	ProjectMeta,
	Repositories,
	Talk,
	TalkLink
} from '$lib/types';
import { loadBib, toPublications, type BibIndex } from './bibtex';
import { renderMarkdown } from './markdown';

const CONTENT_DIR = path.resolve('content');
const POSTS_DIR = path.join(CONTENT_DIR, 'posts');
const NEWS_DIR = path.join(CONTENT_DIR, 'news');
const PROJECTS_DIR = path.join(CONTENT_DIR, 'projects');
const BIBLIOGRAPHY_DIR = path.join(CONTENT_DIR, 'bibliography');

type UnknownRecord = Record<string, unknown>;
type MarkdownDocument = {
	data: UnknownRecord;
	body: string;
};

const markdownCache = new Map<string, MarkdownDocument>();
const postMetaCache = new Map<string, PostMeta>();
const newsMetaCache = new Map<string, Omit<NewsItem, 'html'>>();
const newsCache = new Map<string, NewsItem>();
const projectMetaCache = new Map<string, ProjectMeta>();
const bibliographyCache = new Map<string, BibIndex>();
let publicationsCache: ReturnType<typeof toPublications> | undefined;
let talksCache: Talk[] | undefined;
let cvCache: CVSection[] | undefined;
let repositoriesCache: Repositories | undefined;
let aboutCache: About | undefined;

const VENUE_ICON_MATCHES: Array<{ match: string; icon: string }> = [
	{ match: 'british journal of anaesthesia', icon: '/assets/img/venues/oup.png' },
	{ match: 'addiction biology', icon: '/assets/img/venues/wiley.png' },
	{ match: 'journal of clinical monitoring', icon: '/assets/img/venues/springer.png' },
	{ match: 'aacl', icon: '/assets/img/venues/acl.png' },
	{ match: 'anthology', icon: '/assets/img/venues/acl.png' },
	{ match: 'brain', icon: '/assets/img/venues/oup.png' },
	{ match: 'acl', icon: '/assets/img/venues/acl.png' }
];

function isRecord(value: unknown): value is UnknownRecord {
	return typeof value === 'object' && value !== null;
}

function readText(file: string): string {
	try {
		return readFileSync(file, 'utf8');
	} catch (error) {
		throw new Error(`Unable to read content file ${file}: ${String(error)}`);
	}
}

function readMarkdown(file: string): MarkdownDocument {
	const cached = markdownCache.get(file);
	if (cached) {
		return cached;
	}

	try {
		const parsed = matter(readText(file)) as unknown as { data: unknown; content: string };
		if (!isRecord(parsed.data) || typeof parsed.content !== 'string') {
			throw new Error('frontmatter did not produce an object and body');
		}
		const document = { data: parsed.data, body: parsed.content };
		markdownCache.set(file, document);
		return document;
	} catch (error) {
		throw new Error(`Malformed frontmatter in ${file}: ${String(error)}`);
	}
}

function readYaml(file: string): unknown {
	try {
		return parseYaml(readText(file)) as unknown;
	} catch (error) {
		throw new Error(`Malformed YAML in ${file}: ${String(error)}`);
	}
}

function requiredString(data: UnknownRecord, key: string, file: string): string {
	const value = data[key];
	if (typeof value !== 'string' || value.length === 0) {
		throw new Error(`Expected "${key}" to be a non-empty string in ${file}`);
	}
	return value;
}

function optionalString(data: UnknownRecord, key: string, file: string): string | undefined {
	const value = data[key];
	if (value === undefined || value === null) {
		return undefined;
	}
	if (typeof value !== 'string') {
		throw new Error(`Expected "${key}" to be a string in ${file}`);
	}
	return value;
}

function booleanValue(data: UnknownRecord, key: string, file: string, fallback?: boolean): boolean {
	const value = data[key];
	if (value === undefined && fallback !== undefined) {
		return fallback;
	}
	if (typeof value !== 'boolean') {
		throw new Error(`Expected "${key}" to be a boolean in ${file}`);
	}
	return value;
}

function numberValue(data: UnknownRecord, key: string, file: string): number {
	const value = data[key];
	if (typeof value !== 'number' || !Number.isFinite(value)) {
		throw new Error(`Expected "${key}" to be a number in ${file}`);
	}
	return value;
}

function isoDate(value: unknown, file: string): string {
	if (typeof value === 'string') {
		const match = /^(\d{4}-\d{2}-\d{2})/.exec(value);
		if (match) {
			return match[1];
		}
	}
	if (value instanceof Date && !Number.isNaN(value.valueOf())) {
		return value.toISOString().slice(0, 10);
	}
	throw new Error(`Expected "${file}" to contain a valid date`);
}

function stringList(value: unknown, key: string, file: string): string[] {
	if (!Array.isArray(value) || value.some((item) => typeof item !== 'string')) {
		throw new Error(`Expected "${key}" to be a list of strings in ${file}`);
	}
	return value as string[];
}

function authors(value: unknown, file: string): Author[] {
	if (!Array.isArray(value)) {
		throw new Error(`Expected "authors" to be a list in ${file}`);
	}
	return value.map((item, index) => {
		if (!isRecord(item)) {
			throw new Error(`Expected authors[${index}] to be an object in ${file}`);
		}
		const author: Author = { name: requiredString(item, 'name', file) };
		const url = optionalString(item, 'url', file);
		const affiliation = optionalString(item, 'affiliation', file);
		if (url !== undefined) {
			author.url = url;
		}
		if (affiliation !== undefined) {
			author.affiliation = affiliation;
		}
		return author;
	});
}

function assetPath(value: string, file: string): string {
	if (!value) {
		throw new Error(`Expected an asset path in ${file}`);
	}
	return value.startsWith('/') ? value : `/${value}`;
}

function markdownFiles(directory: string): string[] {
	try {
		return readdirSync(directory)
			.filter((file) => file.endsWith('.md'))
			.map((file) => path.join(directory, file));
	} catch (error) {
		throw new Error(`Unable to list content directory ${directory}: ${String(error)}`);
	}
}

function postSlug(file: string): { year: string; slug: string } {
	const name = path.basename(file, '.md');
	const match = /^(\d{4})-\d{2}-\d{2}-(.+)$/.exec(name);
	if (!match) {
		throw new Error(`Post filename does not include a date prefix: ${file}`);
	}
	// Slug keeps the filename spelling (underscores included) so the Jekyll URLs still resolve.
	return { year: match[1], slug: match[2] };
}

function postMeta(file: string): PostMeta {
	const cached = postMetaCache.get(file);
	if (cached) {
		return cached;
	}
	const document = readMarkdown(file);
	const { data, body } = document;
	const date = isoDate(data.date, file);
	const { year, slug } = postSlug(file);
	const authorsValue = authors(data.authors, file);
	const thumbnail = optionalString(data, 'thumbnail', file);
	const meta: PostMeta = {
		slug,
		year: date.slice(0, 4) || year,
		url: `/blog/${date.slice(0, 4)}/${slug}/`,
		title: requiredString(data, 'title', file),
		description: requiredString(data, 'description', file),
		date,
		tags: stringList(data.tags, 'tags', file),
		authors: authorsValue,
		featured: booleanValue(data, 'featured', file, false),
		readingTime: readingTime(body).text,
		giscus: booleanValue(data, 'giscus', file, false)
	};
	if (thumbnail !== undefined) {
		meta.thumbnail = assetPath(thumbnail, file);
	}
	postMetaCache.set(file, meta);
	return meta;
}

function postFileFor(year: string, slug: string): string | undefined {
	return markdownFiles(POSTS_DIR).find((file) => {
		const parsed = postSlug(file);
		return parsed.year === year && parsed.slug === slug;
	});
}

function bibliographyFor(filename: string, postFile: string): BibIndex {
	const bibliographyFile = path.join(BIBLIOGRAPHY_DIR, filename);
	if (!existsSync(bibliographyFile)) {
		throw new Error(`Missing bibliography ${bibliographyFile} referenced by ${postFile}`);
	}
	const cached = bibliographyCache.get(bibliographyFile);
	if (cached) {
		return cached;
	}
	const index = loadBib(bibliographyFile);
	bibliographyCache.set(bibliographyFile, index);
	return index;
}

export function listPosts(): PostMeta[] {
	return markdownFiles(POSTS_DIR)
		.map(postMeta)
		.sort(
			(left, right) => right.date.localeCompare(left.date) || left.slug.localeCompare(right.slug)
		);
}

export async function loadPost(year: string, slug: string): Promise<Post | null> {
	const file = postFileFor(year, slug);
	if (!file) {
		return null;
	}
	const document = readMarkdown(file);
	const meta = postMeta(file);
	const bibliographyName = optionalString(document.data, 'bibliography', file);
	const rendered = await renderMarkdown(
		document.body,
		bibliographyName ? { bib: bibliographyFor(bibliographyName, file) } : undefined
	);
	return { ...meta, ...rendered };
}

export function listTags(): { tag: string; count: number }[] {
	const counts = new Map<string, number>();
	for (const post of listPosts()) {
		for (const tag of post.tags) {
			counts.set(tag, (counts.get(tag) ?? 0) + 1);
		}
	}
	return [...counts.entries()]
		.map(([tag, count]) => ({ tag, count }))
		.sort((left, right) => right.count - left.count || left.tag.localeCompare(right.tag));
}

function newsItem(file: string): Omit<NewsItem, 'html'> {
	const cached = newsMetaCache.get(file);
	if (cached) {
		return cached;
	}
	const document = readMarkdown(file);
	const slug = path.basename(file, '.md');
	const item: Omit<NewsItem, 'html'> = {
		slug,
		url: `/news/${slug}/`,
		title: requiredString(document.data, 'title', file),
		date: isoDate(document.data.date, file),
		inline: booleanValue(document.data, 'inline', file)
	};
	newsMetaCache.set(file, item);
	return item;
}

export async function listNews(): Promise<NewsItem[]> {
	const items = await Promise.all(
		markdownFiles(NEWS_DIR).map(async (file) => {
			const item = newsItem(file);
			const cached = newsCache.get(file);
			if (cached?.html !== undefined) {
				return cached;
			}
			const rendered = await renderMarkdown(readMarkdown(file).body);
			const complete = { ...item, html: rendered.html };
			newsCache.set(file, complete);
			return complete;
		})
	);
	return items.sort((left, right) => right.date.localeCompare(left.date));
}

function projectMeta(file: string): ProjectMeta {
	const cached = projectMetaCache.get(file);
	if (cached) {
		return cached;
	}
	const { data } = readMarkdown(file);
	const slug = path.basename(file, '.md');
	const meta: ProjectMeta = {
		slug,
		url: `/projects/${slug}/`,
		title: requiredString(data, 'title', file),
		description: requiredString(data, 'description', file),
		importance: numberValue(data, 'importance', file),
		category: requiredString(data, 'category', file)
	};
	const image = optionalString(data, 'img', file);
	const redirect = optionalString(data, 'redirect', file);
	if (image !== undefined) {
		meta.img = assetPath(image, file);
	}
	if (redirect !== undefined) {
		meta.redirect = redirect;
	}
	projectMetaCache.set(file, meta);
	return meta;
}

export function listProjects(): ProjectMeta[] {
	return markdownFiles(PROJECTS_DIR)
		.map(projectMeta)
		.sort(
			(left, right) => left.importance - right.importance || left.title.localeCompare(right.title)
		);
}

export async function loadProject(slug: string): Promise<Project | null> {
	const file = markdownFiles(PROJECTS_DIR).find(
		(candidate) => path.basename(candidate, '.md') === slug
	);
	if (!file) {
		return null;
	}
	const meta = projectMeta(file);
	const rendered = await renderMarkdown(readMarkdown(file).body);
	return { ...meta, html: rendered.html };
}

export function listPublications(): Publication[] {
	if (!publicationsCache) {
		const file = path.join(BIBLIOGRAPHY_DIR, 'papers.bib');
		if (!existsSync(file)) {
			throw new Error(`Missing bibliography ${file}`);
		}
		publicationsCache = toPublications(loadBib(file)).map((publication) => {
			const venue = publication.venueHtml.replace(/<[^>]*>/g, '').toLowerCase();
			const venueIcon = VENUE_ICON_MATCHES.find(({ match }) => venue.includes(match))?.icon;
			return venueIcon === undefined ? publication : { ...publication, venueIcon };
		});
	}
	return publicationsCache;
}

function talkLink(value: unknown, file: string, index: number): TalkLink {
	if (!isRecord(value)) {
		throw new Error(`Expected links[${index}] to be an object in ${file}`);
	}
	const link: TalkLink = {
		name: requiredString(value, 'name', file),
		url: requiredString(value, 'url', file)
	};
	const icon = optionalString(value, 'icon', file);
	if (icon !== undefined) {
		link.icon = icon;
	}
	return link;
}

function talk(value: unknown, file: string, index: number): Talk {
	if (!isRecord(value)) {
		throw new Error(`Expected talks[${index}] to be an object in ${file}`);
	}
	const linksValue = value.links ?? [];
	if (!Array.isArray(linksValue)) {
		throw new Error(`Expected talks[${index}].links to be a list in ${file}`);
	}
	const year = value.year;
	if (typeof year !== 'number' || !Number.isInteger(year)) {
		throw new Error(`Expected talks[${index}].year to be an integer in ${file}`);
	}
	const description = optionalString(value, 'description', file);
	const image = optionalString(value, 'image', file);
	const icon = optionalString(value, 'icon', file);
	const result: Talk = {
		title: requiredString(value, 'title', file),
		kind: requiredString(value, 'kind', file),
		date: requiredString(value, 'date', file),
		year,
		venue: requiredString(value, 'venue', file),
		links: linksValue.map((link, linkIndex) => talkLink(link, file, linkIndex))
	};
	if (description !== undefined) {
		result.description = description;
	}
	if (image !== undefined) {
		result.image = assetPath(image, file);
	}
	if (icon !== undefined) {
		result.icon = assetPath(icon, file);
	}
	return result;
}

export function listTalks(): Talk[] {
	if (!talksCache) {
		const file = path.join(CONTENT_DIR, 'talks.yml');
		const value = readYaml(file);
		if (!Array.isArray(value)) {
			throw new Error(`Expected talks YAML to be a list in ${file}`);
		}
		talksCache = value.map((item, index) => talk(item, file, index));
		talksCache.sort((left, right) => right.year - left.year);
	}
	return talksCache;
}

function cvEntry(value: unknown, file: string, index: number): CVEntry {
	if (!isRecord(value)) {
		throw new Error(`Expected CV contents[${index}] to be an object in ${file}`);
	}
	const entry: CVEntry = {};
	for (const key of ['title', 'institution', 'year', 'name', 'value']) {
		const item = value[key];
		if (item !== undefined) {
			if (typeof item !== 'string' && !(key === 'year' && typeof item === 'number')) {
				throw new Error(`Expected CV ${key} to be a string in ${file}`);
			}
			entry[key as 'title' | 'institution' | 'year' | 'name' | 'value'] = String(item);
		}
	}
	for (const key of ['description', 'items']) {
		const item = value[key];
		if (item !== undefined) {
			if (!Array.isArray(item) || item.some((line) => typeof line !== 'string')) {
				throw new Error(`Expected CV ${key} to be a list of strings in ${file}`);
			}
			entry[key as 'description' | 'items'] = item as string[];
		}
	}
	return entry;
}

export function loadCV(): CVSection[] {
	if (!cvCache) {
		const file = path.join(CONTENT_DIR, 'cv.yml');
		const value = readYaml(file);
		if (!Array.isArray(value)) {
			throw new Error(`Expected CV YAML to be a list in ${file}`);
		}
		const allowedTypes = new Set(['map', 'time_table', 'nested_list', 'list']);
		cvCache = value.map((item, index) => {
			if (!isRecord(item)) {
				throw new Error(`Expected CV section ${index} to be an object in ${file}`);
			}
			const type = requiredString(item, 'type', file);
			if (!allowedTypes.has(type)) {
				throw new Error(`Unknown CV section type "${type}" in ${file}`);
			}
			const contents = item.contents;
			if (!Array.isArray(contents)) {
				throw new Error(`Expected CV section ${index}.contents to be a list in ${file}`);
			}
			return {
				title: requiredString(item, 'title', file),
				type,
				contents: contents.map((entry, entryIndex) => cvEntry(entry, file, entryIndex))
			};
		});
	}
	return cvCache;
}

export function loadRepositories(): Repositories {
	if (!repositoriesCache) {
		const file = path.join(CONTENT_DIR, 'repositories.yml');
		const value = readYaml(file);
		if (!isRecord(value)) {
			throw new Error(`Expected repositories YAML to be an object in ${file}`);
		}
		const users = value.github_users;
		const repos = value.github_repos;
		if (!Array.isArray(users) || users.some((item) => typeof item !== 'string')) {
			throw new Error(`Expected github_users to be a list of strings in ${file}`);
		}
		if (!Array.isArray(repos) || repos.some((item) => typeof item !== 'string')) {
			throw new Error(`Expected github_repos to be a list of strings in ${file}`);
		}
		repositoriesCache = { users: users as string[], repos: repos as string[] };
	}
	return repositoriesCache;
}

export async function loadAbout(): Promise<About> {
	if (aboutCache) {
		return aboutCache;
	}
	const file = path.join(CONTENT_DIR, 'about.md');
	const document = readMarkdown(file);
	const profileValue = document.data.profile;
	if (!isRecord(profileValue)) {
		throw new Error(`Expected profile to be an object in ${file}`);
	}
	const profile: AboutProfile = {
		image: assetPath(requiredString(profileValue, 'image', file), file),
		circular: booleanValue(profileValue, 'circular', file)
	};
	const moreInfo = optionalString(profileValue, 'more_info', file);
	if (moreInfo !== undefined) {
		profile.moreInfoHtml = moreInfo;
	}
	const rendered = await renderMarkdown(document.body);
	aboutCache = {
		subtitleHtml: requiredString(document.data, 'subtitle', file),
		profile,
		html: rendered.html
	};
	return aboutCache;
}
