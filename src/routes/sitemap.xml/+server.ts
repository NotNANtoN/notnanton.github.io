import { site } from '$lib/site';
import {
	listNews,
	listPosts,
	listProjects,
	listTags
} from '$lib/server/content';
import { buildSitemap } from '$lib/server/feed';
import type { RequestHandler } from './$types';

export const prerender = true;

export const GET: RequestHandler = async () => {
	const posts = listPosts();
	const news = await listNews();
	const urls = [
		'/',
		'/blog/',
		'/publications/',
		'/projects/',
		'/talks/',
		'/cv/',
		'/repositories/',
		'/news/',
		...posts.map((post) => post.url),
		...new Set(posts.map((post) => `/blog/${post.year}/`)),
		...listTags().map(({ tag }) => `/blog/tag/${encodeURIComponent(tag)}/`),
		...listProjects().map((project) => project.url),
		...news.map((item) => item.url)
	].map((path) => `${site.url}${path}`);

	return new Response(buildSitemap([...new Set(urls)]), {
		headers: {
			'Content-Type': 'application/xml; charset=utf-8',
			'Cache-Control': 'public, max-age=3600'
		}
	});
};
