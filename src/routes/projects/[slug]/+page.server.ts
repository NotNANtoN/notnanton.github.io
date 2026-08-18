import { error } from '@sveltejs/kit';
import { listProjects, loadProject } from '$lib/server/content';
import type { EntryGenerator, PageServerLoad } from './$types';

export const entries: EntryGenerator = () => listProjects().map((project) => ({ slug: project.slug }));

export const load: PageServerLoad = async ({ params }) => {
	const project = await loadProject(params.slug);
	if (!project) error(404, 'Project not found');
	return { project };
};
