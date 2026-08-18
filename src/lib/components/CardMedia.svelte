<script lang="ts">
	interface Props {
		src: string;
		alt?: string;
		ratio?: string;
		class?: string;
	}

	let { src, alt = '', ratio = '16/9', class: className = '' }: Props = $props();
</script>

<div class={['relative overflow-hidden', className]} style={`aspect-ratio: ${ratio};`}>
	<img
		class="absolute inset-0 h-full w-full scale-110 object-cover opacity-60 blur-2xl saturate-[1.15]"
		src={src}
		alt=""
		aria-hidden="true"
		loading="lazy"
		decoding="async"
	/>
	<img
		class="relative z-10 h-full w-full object-contain"
		src={src}
		alt={alt}
		loading="lazy"
		decoding="async"
	/>
</div>
