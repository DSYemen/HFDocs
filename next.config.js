const withNextra = require('nextra')({
  theme: 'nextra-theme-docs',
  themeConfig: './theme.config.tsx',
  distDir: 'public',
})

module.exports = withNextra()
