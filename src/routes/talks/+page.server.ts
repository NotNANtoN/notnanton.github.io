import { listTalks } from '$lib/server/content';
import type { PageServerLoad } from './$types';

export const load: PageServerLoad = () => {
	const groups = new Map<number, ReturnType<typeof listTalks>>();
	for (const talk of listTalks()) {
		const group = groups.get(talk.year) ?? [];
		group.push(talk);
		groups.set(talk.year, group);
	}

	return {
		groups: [...groups.entries()]
			.sort(([yearA], [yearB]) => yearB - yearA)
			.map(([year, talks]) => ({ year, talks }))
	};
};
