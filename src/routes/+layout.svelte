<script lang="ts">
	import '../app.css';
	import type { Snippet } from 'svelte';
	import Footer from '$lib/components/Footer.svelte';
	import Header from '$lib/components/Header.svelte';

	let { children }: { children: Snippet } = $props();
</script>

<Header />
<main class="min-h-[calc(100vh-4rem)] pt-12">
	{@render children()}
</main>
<Footer />
