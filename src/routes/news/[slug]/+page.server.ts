import { error } from '@sveltejs/kit';
import { listNews } from '$lib/server/content';
import type { EntryGenerator, PageServerLoad } from './$types';

export const entries: EntryGenerator = async () =>
	(await listNews()).map((item) => ({ slug: item.slug }));

export const load: PageServerLoad = async ({ params }) => {
	const item = (await listNews()).find((newsItem) => newsItem.slug === params.slug);
	if (!item) error(404, 'News item not found');
	return { item };
};
