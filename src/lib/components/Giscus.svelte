<script lang="ts">
	import { giscus } from '$lib/site';

	let { enabled, title }: { enabled: boolean; title: string } = $props();

	function attachGiscus(container: HTMLDivElement) {
		const script = document.createElement('script');
		script.src = 'https://giscus.app/client.js';
		script.async = true;
		script.crossOrigin = 'anonymous';
		script.setAttribute('data-repo', giscus.repo);
		script.setAttribute('data-repo-id', giscus.repoId);
		script.setAttribute('data-category', giscus.category);
		script.setAttribute('data-category-id', giscus.categoryId);
		script.setAttribute('data-mapping', giscus.mapping);
		script.setAttribute('data-strict', giscus.strict);
		script.setAttribute('data-reactions-enabled', giscus.reactionsEnabled);
		script.setAttribute('data-input-position', giscus.inputPosition);
		script.setAttribute('data-theme', document.documentElement.classList.contains('dark') ? 'dark' : 'light');
		script.setAttribute('data-lang', 'en');
		container.append(script);
		const themeObserver = new MutationObserver(() => {
			const theme = document.documentElement.classList.contains('dark') ? 'dark' : 'light';
			script.setAttribute('data-theme', theme);
			container
				.querySelector<HTMLIFrameElement>('iframe.giscus-frame')
				?.contentWindow?.postMessage({ giscus: { setConfig: { theme } } }, 'https://giscus.app');
		});
		themeObserver.observe(document.documentElement, { attributes: true, attributeFilter: ['class'] });
		return () => {
			themeObserver.disconnect();
			script.remove();
		};
	}
</script>

{#if enabled && giscus.enabled}
	<section class="mt-12 border-t border-border pt-8" aria-label="Comments">
		<h2 class="text-xl font-semibold">Comments</h2>
		<div {@attach attachGiscus} data-giscus-title={title}></div>
	</section>
{/if}
