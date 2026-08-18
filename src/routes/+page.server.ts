import { homeSections } from '$lib/site';
import {
	listNews,
	listPosts,
	listPublications,
	loadAbout
} from '$lib/server/content';
import type { PageServerLoad } from './$types';

export const load: PageServerLoad = async () => {
	const posts = listPosts();
	const publications = listPublications();

	return {
		about: await loadAbout(),
		news: (await listNews()).slice(0, homeSections.newsLimit),
		posts: posts.slice(0, homeSections.latestPostsLimit),
		selectedPublications: publications.filter((publication) => publication.selected)
	};
};
