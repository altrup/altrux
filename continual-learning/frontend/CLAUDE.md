# Claude Guidelines — continual-learning/frontend

## Quality checks (run in this order)

After completing any change, run these commands in sequence:

1. `npm run lint` — fix all ESLint errors before continuing
2. `npm run typecheck` — fix all TypeScript errors before continuing
3. `npm run format` — run Prettier last, after all errors are resolved

Never skip steps or reorder them. Format only after lint and typecheck are clean.

## Tailwind colours

Never hardcode colour values in JSX or CSS (no `text-[#abc]`, no `bg-[rgba(...)]`, no inline `style` colour properties).

All colours must be declared as CSS custom properties inside the `@theme` block in `app/app.css`, then referenced by their Tailwind utility name.

**Adding a new colour:**

```css
/* app/app.css */
@theme {
  --color-primary: #6366f1;
  --color-surface: #1e1e2e;
}
```

This automatically exposes `bg-primary`, `text-primary`, `border-primary`, `bg-surface`, etc. via Tailwind v4's CSS-variable-driven theme — no `tailwind.config` edits needed.

Use only custom colour names in JSX — never Tailwind's built-in palette:

```tsx
<div className="bg-surface text-primary" />      // ✅
<div className="bg-neutral-900 text-gray-100" /> // ❌ built-in palette
<div style={{ background: '#1e1e2e' }} />         // ❌ hardcoded
<div className="bg-[#1e1e2e]" />                 // ❌ arbitrary value
```

If a colour is needed and doesn't have a variable yet, add one to `app/app.css` first.

Keep all colour variables grouped at the top of the `@theme` block with a comment, before font and other tokens.
