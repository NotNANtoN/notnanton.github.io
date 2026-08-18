<script lang="ts">
	import { site } from '$lib/site';

	let {
		title,
		description = site.description,
		canonicalPath
	}: {
		title: string;
		description?: string;
		canonicalPath: string;
	} = $props();

	const canonical = $derived(`${site.url}${canonicalPath}`);
</script>

<svelte:head>
	<title>{title} · {site.shortName}</title>
	<meta name="description" content={description} />
	<link rel="canonical" href={canonical} />
	<link rel="alternate" type="application/rss+xml" title={site.blogName} href={`${site.url}/feed.xml`} />
	<meta property="og:title" content={`${title} · ${site.shortName}`} />
	<meta property="og:description" content={description} />
	<meta property="og:url" content={canonical} />
	<meta property="og:type" content="website" />
	<meta name="twitter:card" content="summary" />
	<meta name="twitter:title" content={`${title} · ${site.shortName}`} />
	<meta name="twitter:description" content={description} />
</svelte:head>
