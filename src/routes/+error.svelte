<script lang="ts">
	import CircleAlert from '~icons/lucide/circle-alert';
	import Home from '~icons/lucide/home';
	import { page } from '$app/state';
	import SiteHead from '$lib/components/SiteHead.svelte';

	const status = $derived(page.status);
	const message = $derived(page.error?.message ?? 'This page does not exist or may have moved.');
</script>

<SiteHead title={`${status}`} description="The requested page could not be found." canonicalPath="/" />

<div class="page-container flex min-h-[55vh] flex-col items-center justify-center text-center">
	<CircleAlert class="mb-5 text-accent" width={48} height={48} stroke-width={1.5} />
	<p class="text-sm font-semibold uppercase tracking-[0.2em] text-muted">{status}</p>
	<h1 class="mt-3 text-4xl font-bold tracking-tight">Page not found</h1>
	<p class="mt-4 max-w-md text-muted">{message}</p>
	<a class="mt-7 inline-flex items-center gap-2 rounded-lg bg-accent px-4 py-2.5 font-semibold text-white no-underline hover:bg-accent-hover" href="/">
		<Home width={17} height={17} />Return home
	</a>
</div>
