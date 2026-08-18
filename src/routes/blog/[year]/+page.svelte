<script lang="ts">
	import CalendarDays from '~icons/lucide/calendar-days';
	import type { PageProps } from './$types';
	import SiteHead from '$lib/components/SiteHead.svelte';
	import { formatDate } from '$lib/components/format';

	let { data }: PageProps = $props();
</script>

<SiteHead title={`Posts from ${data.year}`} description={`An archive of posts from ${data.year}.`} canonicalPath={`/blog/${data.year}/`} />

<div class="page-container">
	<header class="mb-10">
		<a class="mb-4 inline-flex items-center gap-1 text-sm font-semibold no-underline" href="/blog/">← All posts</a>
		<h1 class="flex items-center gap-3 text-4xl font-bold tracking-tight"><CalendarDays class="text-accent" width={32} height={32} />{data.year}</h1>
		<p class="mt-3 text-muted">An archive of posts from this year.</p>
	</header>

	<div class="overflow-hidden rounded-xl border border-border">
		<table class="w-full text-left text-sm">
			<tbody class="divide-y divide-border">
				{#each data.posts as post (post.url)}
					<tr class="hover:bg-surface/60">
						<th class="whitespace-nowrap px-4 py-4 font-medium text-muted sm:w-36" scope="row">{formatDate(post.date)}</th>
						<td class="px-4 py-4"><a class="font-semibold no-underline" href={post.url}>{post.title}</a></td>
					</tr>
				{:else}
					<tr><td class="px-4 py-4 text-muted">No posts found.</td></tr>
				{/each}
			</tbody>
		</table>
	</div>
</div>
