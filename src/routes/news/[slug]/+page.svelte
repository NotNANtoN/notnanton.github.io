<script lang="ts">
	import CalendarDays from '~icons/lucide/calendar-days';
	import type { PageProps } from './$types';
	import SiteHead from '$lib/components/SiteHead.svelte';
	import { formatDate } from '$lib/components/format';

	let { data }: PageProps = $props();
</script>

<SiteHead title={data.item.title} description={`News: ${data.item.title}`} canonicalPath={data.item.url} />

<div class="page-container">
	<a class="mb-8 inline-flex items-center gap-1 text-sm font-semibold no-underline" href="/news/">← All news</a>
	<header class="mb-10">
		<time class="inline-flex items-center gap-1.5 text-sm text-muted" datetime={data.item.date}><CalendarDays width={16} height={16} />{formatDate(data.item.date)}</time>
		<h1 class="mt-3 text-4xl font-bold leading-tight tracking-tight">{data.item.title}</h1>
	</header>
	<article class="prose prose-neutral max-w-none dark:prose-invert">
		{@html data.item.html}
	</article>
</div>
