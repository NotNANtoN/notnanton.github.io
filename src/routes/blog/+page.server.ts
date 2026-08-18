import { listPosts, listTags } from '$lib/server/content';
import type { PageServerLoad } from './$types';

export const load: PageServerLoad = () => ({
	posts: listPosts(),
	tags: listTags()
});
