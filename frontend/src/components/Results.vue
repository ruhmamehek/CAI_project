<template>
  <section class="results-section">
    <div class="answer-box">
      <h2>Answer</h2>
      <div class="answer-content" ref="answerContent" v-html="formattedAnswer"></div>
    </div>
    
    <div 
      v-if="tooltip.visible" 
      class="custom-tooltip"
      :style="{ top: tooltip.y + 'px', left: tooltip.x + 'px' }"
    >
      {{ tooltip.text }}
    </div>

    <div class="sources-box">
      <h2>Sources ({{ result.num_chunks_retrieved }})</h2>
      <div class="sources-list">
        <div
          v-for="(source, index) in result.sources"
          :key="index"
          class="source-item"
        >
          <div class="source-header">
            <div class="source-header-left">
              <span class="source-ticker">{{ source.ticker }}</span>
              <span class="source-year">{{ source.year }}</span>
              <span class="source-chunk-id">{{ source.chunk_id }}</span>
              <span class="source-score">Score: {{ source.score.toFixed(3) }}</span>
            </div>
            <button 
              v-if="source.text"
              @click="toggleSource(index)"
              class="source-expand-btn"
              :aria-expanded="isExpanded(index)"
              :title="isExpanded(index) ? 'Hide source text' : 'Show source text'"
            >
              <span class="expand-icon" :class="{ 'expanded': isExpanded(index) }">▼</span>
            </button>
          </div>
          <div class="source-details">
            <small>Chunk ID: {{ source.chunk_id }}</small>
            <small v-if="source.chunk_type !== 'Text'"> | Type: {{ source.chunk_type }}</small>
            <small v-if="source.page"> | Page: {{ source.page }}</small>
          </div>
          <div v-if="source.image_path" class="source-image-container">
            <img 
              :src="getImageUrl(source.image_path)" 
              :alt="`${source.chunk_type} from ${source.ticker} ${source.year}`"
              class="source-image"
              @error="handleImageError"
            />
          </div>
          <div 
            v-if="source.text && isExpanded(index)" 
            class="source-text-content"
          >
            <div class="source-text">{{ source.text }}</div>
          </div>
        </div>
      </div>
    </div>
  </section>
</template>

<script>
export default {
  name: 'Results',
  props: {
    result: {
      type: Object,
      required: true
    }
  },
  data() {
    return {
      tooltip: {
        visible: false,
        text: '',
        x: 0,
        y: 0
      },
      expandedSources: new Set()
    }
  },
  computed: {
    formattedAnswer() {
      if (!this.result?.answer) return ''
      
      const sourceTagRegex = /<source\s+([^>]+)>([^<]+)<\/source>/gi
      
      let formatted = this.result.answer
      const parts = []
      let lastIndex = 0
      let match
      
      sourceTagRegex.lastIndex = 0
      
      while ((match = sourceTagRegex.exec(formatted)) !== null) {
        if (match.index > lastIndex) {
          parts.push({
            type: 'text',
            content: this.escapeHtml(formatted.substring(lastIndex, match.index))
          })
        }
        
        const attrs = {}
        const attrRegex = /(\w+)="([^"]+)"/g
        let attrMatch
        while ((attrMatch = attrRegex.exec(match[1])) !== null) {
          attrs[attrMatch[1]] = attrMatch[2]
        }
        
        parts.push({
          type: 'highlight',
          content: match[2],
          attrs: attrs
        })
        
        lastIndex = sourceTagRegex.lastIndex
      }
      
      if (lastIndex < formatted.length) {
        parts.push({
          type: 'text',
          content: this.escapeHtml(formatted.substring(lastIndex))
        })
      }
      
      return parts.map(part => {
        if (part.type === 'highlight') {
          const attrs = part.attrs
          const chunkId = attrs.chunk_id || 'Unknown'
          return `<span class="source-highlight" 
                        data-ticker="${this.escapeHtml(attrs.ticker || '')}" 
                        data-year="${this.escapeHtml(attrs.year || '')}" 
                        data-chunk-id="${this.escapeHtml(chunkId)}">${this.escapeHtml(part.content)}</span>`
        } else {
          return part.content
        }
      }).join('')
    }
  },
  methods: {
    escapeHtml(text) {
      const div = document.createElement('div')
      div.textContent = text
      return div.innerHTML
    },
    showTooltip(event, chunkId, ticker, year) {
      const tooltipText = `Chunk ID: ${chunkId} | Source: ${ticker} ${year}`
      this.tooltip = {
        visible: true,
        text: tooltipText,
        x: event.clientX + 10,
        y: event.clientY + 10
      }
    },
    hideTooltip() {
      this.tooltip.visible = false
    },
    toggleSource(index) {
      if (this.expandedSources.has(index)) {
        this.expandedSources.delete(index)
      } else {
        this.expandedSources.add(index)
      }
    },
    isExpanded(index) {
      return this.expandedSources.has(index)
    },
    attachTooltipListeners() {
      if (!this.$refs.answerContent) return
      
      const highlights = this.$refs.answerContent.querySelectorAll('.source-highlight')
      highlights.forEach(span => {
        const newSpan = span.cloneNode(true)
        span.parentNode.replaceChild(newSpan, span)
        
        const chunkId = newSpan.getAttribute('data-chunk-id') || 'Unknown'
        const ticker = newSpan.getAttribute('data-ticker') || 'Unknown'
        const year = newSpan.getAttribute('data-year') || ''
        
        newSpan.addEventListener('mouseenter', (e) => {
          this.showTooltip(e, chunkId, ticker, year)
        })
        newSpan.addEventListener('mouseleave', () => {
          this.hideTooltip()
        })
        newSpan.addEventListener('mousemove', (e) => {
          if (this.tooltip.visible) {
            this.tooltip.x = e.clientX + 10
            this.tooltip.y = e.clientY + 10
          }
        })
      })
    },
    getImageUrl(imagePath) {
      const apiBaseUrl = import.meta.env.VITE_API_URL || 'http://localhost:8888'
      if (imagePath.startsWith('http')) {
        return imagePath
      }
      return `${apiBaseUrl}/images/${imagePath}`
    },
    handleImageError(event) {
      console.error('Error loading image:', event.target.src)
      event.target.style.display = 'none'
    }
  },
  mounted() {
    this.$nextTick(() => {
      this.attachTooltipListeners()
    })
  },
  updated() {
    this.$nextTick(() => {
      this.attachTooltipListeners()
    })
  }
}
</script>

