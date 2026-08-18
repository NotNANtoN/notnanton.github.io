import { listPublications } from '$lib/server/content';
import type { PageServerLoad } from './$types';

export const load: PageServerLoad = () => {
	const groups = new Map<number, ReturnType<typeof listPublications>>();
	for (const publication of listPublications()) {
		const group = groups.get(publication.year) ?? [];
		group.push(publication);
		groups.set(publication.year, group);
	}

	return {
		groups: [...groups.entries()]
			.sort(([yearA], [yearB]) => yearB - yearA)
			.map(([year, publications]) => ({ year, publications }))
	};
};
