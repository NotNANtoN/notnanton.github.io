import { listPosts, listTags } from '$lib/server/content';
import type { EntryGenerator, PageServerLoad } from './$types';

export const entries: EntryGenerator = () => listTags().map(({ tag }) => ({ tag }));

export const load: PageServerLoad = ({ params }) => {
	const tag = decodeURIComponent(params.tag);
	return {
		tag,
		posts: listPosts().filter((post) => post.tags.some((item) => item.toLowerCase() === tag.toLowerCase()))
	};
};
