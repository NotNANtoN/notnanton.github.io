<script lang="ts">
	import ExternalLink from '~icons/lucide/external-link';
	import File from '~icons/lucide/file';
	import FileText from '~icons/lucide/file-text';
	import Globe from '~icons/lucide/globe';
	import Presentation from '~icons/lucide/presentation';
	import Users from '~icons/lucide/users';
	import Video from '~icons/lucide/video';
	import type { TalkLink } from '$lib/types';

	let { link }: { link: TalkLink } = $props();
</script>

<a class="icon-link" href={link.url} target="_blank" rel="noreferrer">
	{#if link.icon === 'globe'}
		<Globe width={14} height={14} />
	{:else if link.icon === 'pdf'}
		<FileText width={14} height={14} />
	{:else if link.icon === 'video'}
		<Video width={14} height={14} />
	{:else if link.icon === 'users'}
		<Users width={14} height={14} />
	{:else if link.icon === 'slides'}
		<Presentation width={14} height={14} />
	{:else}
		<File width={14} height={14} />
	{/if}
	{link.name}
	<ExternalLink class="opacity-50" width={12} height={12} />
</a>
