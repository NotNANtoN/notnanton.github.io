import { loadRepositories } from '$lib/server/content';
import type { PageServerLoad } from './$types';

export const load: PageServerLoad = () => ({
	repositories: loadRepositories()
});
