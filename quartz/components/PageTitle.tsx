import { joinSegments, pathToRoot } from "../util/path"
import { QuartzComponent, QuartzComponentConstructor, QuartzComponentProps } from "./types"
import { classNames } from "../util/lang"
import { i18n } from "../i18n"

const PageTitle: QuartzComponent = ({ fileData, cfg, displayClass }: QuartzComponentProps) => {
  const title = cfg?.pageTitle ?? i18n(cfg.locale).propertyDefaults.title
  const baseDir = pathToRoot(fileData.slug!)
  const profileImgPath = joinSegments(baseDir, "static/profile.jpg")
  return (
    <div class={classNames(displayClass, "page-title-container")}>
      <a href={baseDir} class="page-title-link">
        <img src={profileImgPath} alt={title} class="page-title-avatar" />
        <h2 class="page-title">{title}</h2>
      </a>
    </div>
  )
}

PageTitle.css = `
.page-title-container {
  margin: 0;
}

.page-title-link {
  display: flex;
  flex-direction: column;
  align-items: flex-start;
  gap: 0.75rem;
  text-decoration: none;
}

.page-title-avatar {
  width: 110px;
  height: 110px;
  border-radius: 14px;
  object-fit: cover;
  margin: 0 !important;
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
  border: 2px solid var(--lightgray);
  transition: transform 0.2s ease, border-color 0.2s ease;
}

.page-title-link:hover .page-title-avatar {
  transform: scale(1.03);
  border-color: var(--secondary);
}

.page-title {
  font-size: 1.55rem;
  line-height: 1.25;
  margin: 0;
  font-family: var(--titleFont);
}

@media all and (max-width: 800px) {
  .page-title-link {
    flex-direction: row;
    align-items: center;
    gap: 0.6rem;
  }
  .page-title-avatar {
    width: 40px;
    height: 40px;
    border-radius: 8px;
  }
  .page-title {
    font-size: 1.2rem;
  }
}
`

export default (() => PageTitle) satisfies QuartzComponentConstructor
