// The site's pages. gated pages show Brian's own squad and sit behind the login; everything else
// is public. server/routes.js keeps its own allow-list of public paths: a page missing
// from that list needs a login, so forgetting to add one fails safe.
export interface NavItem { label: string; href: string; gated: boolean }

export const NAV: NavItem[] = [
  { label: "Players", href: "/", gated: false },
  { label: "Fixtures", href: "/fixtures/", gated: false },
  { label: "Optimal Squad", href: "/optimal-squad/", gated: false },
  { label: "Model Lab", href: "/model-lab/", gated: false },
  { label: "This Week", href: "/this-week/", gated: true },
  { label: "Season", href: "/season/", gated: true },
];
