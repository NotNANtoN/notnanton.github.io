import { listPosts } from '$lib/server/content';
import type { EntryGenerator, PageServerLoad } from './$types';

export const entries: EntryGenerator = () =>
	[...new Set(listPosts().map((post) => post.year))].map((year) => ({ year }));

export const load: PageServerLoad = ({ params }) => {
	const posts = listPosts().filter((post) => post.year === params.year);
	return { year: params.year, posts };
};
