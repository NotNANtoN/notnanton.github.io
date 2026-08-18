/**
 * Shared content contract. The server-side loaders in `src/lib/server/` produce these
 * shapes; routes and components consume them. Keep in sync on both sides.
 */

export type Author = {
	name: string;
	url?: string;
	affiliation?: string;
};

export type TocItem = {
	id: string;
	title: string;
	depth: number;
};

/** A bibliography reference cited from a post via <d-cite key="...">. */
export type Reference = {
	key: string;
	/** 1-based citation number, in order of first appearance in the post. */
	index: number;
	/** Formatted reference, may contain inline markup (<em>, <a>). */
	html: string;
	url?: string;
};

export type PostMeta = {
	slug: string;
	/** Publication year, used for the URL. */
	year: string;
	/** Canonical URL with trailing slash, e.g. /blog/2026/robot-learning-setup/ */
	url: string;
	title: string;
	description: string;
	/** ISO 8601 date string. */
	date: string;
	tags: string[];
	authors: Author[];
	featured: boolean;
	/** Absolute path under /assets, e.g. /assets/img/robot-cube-thumb.jpg */
	thumbnail?: string;
	/** Human readable, e.g. "12 min read". */
	readingTime: string;
	giscus: boolean;
};

export type Post = PostMeta & {
	/** Rendered post body (sanitised at authoring time, trusted content). */
	html: string;
	toc: TocItem[];
	/** Only references actually cited, ordered by citation index. */
	references: Reference[];
};

export type NewsItem = {
	slug: string;
	url: string;
	title: string;
	date: string;
	/** Inline items render in place; non-inline items link to their own page. */
	inline: boolean;
	html: string;
};

export type ProjectMeta = {
	slug: string;
	url: string;
	title: string;
	description: string;
	/** Absolute path under /assets. */
	img?: string;
	/** Lower value sorts first. */
	importance: number;
	category: string;
	/** If set, the project page redirects here instead of rendering a body. */
	redirect?: string;
};

export type Project = ProjectMeta & {
	html: string;
};

export type PublicationLink = {
	/** Label shown on the button, e.g. "arXiv", "PDF", "Code". */
	name: string;
	url: string;
};

export type Publication = {
	key: string;
	/** BibTeX entry type, e.g. article, inproceedings, mastersthesis. */
	type: string;
	title: string;
	/** Rendered author names in order, e.g. "Anton O. Wiehe". */
	authors: string[];
	/** Venue line with markup, e.g. "<em>Brain</em>, 145(8), 2910-2919". */
	venueHtml: string;
	year: number;
	/** Short venue badge, e.g. "BRAIN". */
	abbr?: string;
	venueIcon?: string;
	selected: boolean;
	/** Raw BibTeX for the copy/show button. */
	bibtex: string;
	links: PublicationLink[];
};

export type TalkLink = {
	name: string;
	url: string;
	/** Icon hint: globe | pdf | video | users | file | slides */
	icon?: string;
};

export type Talk = {
	title: string;
	/** Talk | Poster | Paper | Science Slam */
	kind: string;
	/** Display date, e.g. "January 2026". */
	date: string;
	year: number;
	venue: string;
	description?: string;
	image?: string;
	icon?: string;
	links: TalkLink[];
};

export type CVEntry = {
	title?: string;
	institution?: string;
	year?: string;
	description?: string[];
	items?: string[];
	name?: string;
	value?: string;
};

export type CVSection = {
	title: string;
	/** map | time_table | nested_list | list */
	type: string;
	contents: CVEntry[];
};

export type AboutProfile = {
	/** Absolute path under /assets. */
	image: string;
	circular: boolean;
	moreInfoHtml?: string;
};

export type About = {
	/** Rendered subtitle, may contain links. */
	subtitleHtml: string;
	profile: AboutProfile;
	html: string;
};

export type Repositories = {
	users: string[];
	repos: string[];
};
