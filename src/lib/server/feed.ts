import { site } from '$lib/site';
import type { PostMeta } from '$lib/types';

function escapeXml(value: string): string {
	return value
		.replace(/&/g, '&amp;')
		.replace(/</g, '&lt;')
		.replace(/>/g, '&gt;')
		.replace(/"/g, '&quot;')
		.replace(/'/g, '&apos;');
}

function absoluteUrl(value: string): string {
	if (/^https?:\/\//i.test(value)) {
		return value;
	}
	return `${site.url.replace(/\/+$/, '')}/${value.replace(/^\/+/, '')}`;
}

function postDate(date: string): string {
	const parsed = new Date(`${date}T00:00:00Z`);
	return Number.isNaN(parsed.valueOf()) ? date : parsed.toUTCString();
}

export function buildFeed(posts: PostMeta[]): string {
	const channelLink = absoluteUrl('/blog/');
	const items = posts
		.map((post) => {
			const link = absoluteUrl(post.url);
			return [
				'<item>',
				`<title>${escapeXml(post.title)}</title>`,
				`<link>${escapeXml(link)}</link>`,
				`<guid isPermaLink="true">${escapeXml(link)}</guid>`,
				`<pubDate>${escapeXml(postDate(post.date))}</pubDate>`,
				`<description>${escapeXml(post.description)}</description>`,
				'</item>'
			].join('');
		})
		.join('');

	return [
		'<?xml version="1.0" encoding="UTF-8"?>',
		'<rss version="2.0">',
		'<channel>',
		`<title>${escapeXml(site.blogName)}</title>`,
		`<link>${escapeXml(channelLink)}</link>`,
		`<description>${escapeXml(site.blogDescription)}</description>`,
		items,
		'</channel>',
		'</rss>'
	].join('');
}

export function buildSitemap(urls: string[]): string {
	const entries = urls
		.map((url) => `<url><loc>${escapeXml(absoluteUrl(url))}</loc></url>`)
		.join('');
	return [
		'<?xml version="1.0" encoding="UTF-8"?>',
		'<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">',
		entries,
		'</urlset>'
	].join('');
}
