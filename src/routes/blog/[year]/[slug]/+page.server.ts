import { error } from '@sveltejs/kit';
import { listPosts, loadPost } from '$lib/server/content';
import type { EntryGenerator, PageServerLoad } from './$types';

export const entries: EntryGenerator = () =>
	listPosts().map((post) => ({ year: post.year, slug: post.slug }));

export const load: PageServerLoad = async ({ params }) => {
	const posts = listPosts();
	const post = await loadPost(params.year, params.slug);
	if (!post) error(404, 'Post not found');

	const index = posts.findIndex((item) => item.url === post.url);
	return {
		post,
		previous: index >= 0 ? posts[index + 1] : undefined,
		next: index > 0 ? posts[index - 1] : undefined
	};
};
