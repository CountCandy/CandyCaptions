import shutil

from PIL import Image

from candycaption.config import data_dir, load_settings
from candycaption.dataset import Project
from candycaption.imaging import has_transparency, prepare_bytes

from conftest import make_image


def checks(project, name, settings):
    image = project.store.image_by_rel(name)
    return {b["check"] for b in project.health(image, settings)}


def test_health_checks(dataset):
    settings = dict(load_settings(), tokenizer_repo="", token_limit=20)
    make_image(dataset / "tiny.png", size=(200, 200), tags="x")
    make_image(dataset / "wide.png", size=(3000, 600), tags="x")
    make_image(dataset / "clear.png", (5, 5, 5), tags="x", mode="RGBA")
    Image.open(dataset / "clear.png").convert("RGBA").resize((900, 900)).save(dataset / "clear.png")
    img = Image.open(dataset / "clear.png")
    img.putpixel((0, 0), (0, 0, 0, 0))
    img.save(dataset / "clear.png")
    sideways = Image.new("RGB", (1200, 800), (9, 80, 160))
    exif = sideways.getexif()
    exif[0x0112] = 6
    sideways.save(dataset / "sideways.jpg", exif=exif)
    shutil.copy(dataset / "plain.png", dataset / "twin.png")
    Image.open(dataset / "unsure.png").resize((500, 500)).save(dataset / "unsure_small.jpg", quality=80)
    Image.open(dataset / "clash.png").save(dataset / "clash.jpg")
    (dataset / "broken.png").write_bytes(b"not an image")
    (dataset / "plain.txt").write_text(", ".join(f"tag_{n}" for n in range(60)))
    (dataset / "plain_nl.txt").write_text("A caption.")

    project = Project(dataset)
    project.scan()
    assert checks(project, "tiny.png", settings) >= {"too_small", "missing_files"}
    assert "extreme_shape" in checks(project, "wide.png", settings)
    assert "transparency" in checks(project, "clear.png", settings)
    assert "sideways" in checks(project, "sideways.jpg", settings)
    assert "exact_duplicate" in checks(project, "twin.png", settings) and "exact_duplicate" in checks(project, "plain.png", settings)
    assert "near_duplicate" in checks(project, "unsure_small.jpg", settings)
    assert "name_clash" in checks(project, "clash.jpg", settings) and "name_clash" in checks(project, "clash.png", settings)
    assert checks(project, "broken.png", settings) == {"unreadable"}
    assert "too_many_tokens" in checks(project, "plain.png", settings)
    assert "missing_files" in checks(project, "sideways.jpg", settings)   # no tags and, now that captioning has begun, no description


def test_missing_description_is_not_flagged_before_captioning_starts(dataset):
    project = Project(dataset)
    project.scan()
    settings = dict(load_settings(), tokenizer_repo="")
    assert "missing_files" not in checks(project, "plain.png", settings)


def test_fixes_keep_a_backup_and_match_what_the_trainer_sees(dataset):
    Image.new("RGBA", (800, 800), (0, 0, 0, 0)).save(dataset / "clear.png")
    sideways = Image.new("RGB", (1200, 800), (9, 80, 160))
    exif = sideways.getexif()
    exif[0x0112] = 6
    sideways.save(dataset / "sideways.jpg", exif=exif)
    project = Project(dataset)
    project.scan()
    settings = dict(load_settings(), tokenizer_repo="")

    # The captioner already sees transparency filled with the background colour (white by default).
    seen = Image.open(__import__("io").BytesIO(prepare_bytes(dataset / "clear.png", 512, "#ffffff")))
    assert seen.getpixel((10, 10))[0] > 250

    project.fix(project.store.image_by_rel("clear.png"), "transparency")
    fixed = Image.open(dataset / "clear.png")
    assert not has_transparency(fixed) and fixed.convert("RGB").getpixel((10, 10)) == (255, 255, 255)
    project.fix(project.store.image_by_rel("sideways.jpg"), "sideways")
    assert Image.open(dataset / "sideways.jpg").size == (800, 1200)
    assert checks(project, "clear.png", settings) & {"transparency", "sideways"} == set()
    assert checks(project, "sideways.jpg", settings) & {"transparency", "sideways"} == set()
    backups = data_dir() / "backups" / project.id
    assert has_transparency(Image.open(backups / "clear.png")) and Image.open(backups / "sideways.jpg").size == (1200, 800)


def test_store_lives_outside_the_dataset_folder(dataset):
    project = Project(dataset)
    project.scan()
    before = sorted(p.name for p in dataset.iterdir())
    assert all(not name.endswith((".sqlite", ".json")) for name in before)
    assert project.store.path.is_relative_to(data_dir())


def test_find_and_replace_previews_then_keeps_history(dataset):
    project = Project(dataset)
    project.scan()
    for name in ("plain", "unsure"):
        (dataset / f"{name}_nl.txt").write_text("A girl stands. A Girl waves at the girlfriend.")
    changes = project.replace_preview("girl", "woman", whole=True)
    assert len(changes) == 2 and changes[0]["after"] == "A woman stands. A woman waves at the girlfriend."
    assert (dataset / "plain_nl.txt").read_text().startswith("A girl")       # preview writes nothing
    assert project.replace_preview("girl", "woman", whole=True, case=True)[0]["after"] == "A woman stands. A Girl waves at the girlfriend."
    assert project.replace_preview(r"A (\w+) stands\. ", "", regex=True)[0]["after"] == "A Girl waves at the girlfriend."
    assert project.replace_apply(changes, only={changes[0]["id"]}) == 1
    image = project.store.image(changes[0]["id"])
    assert project.read_caption(image).startswith("A woman")
    assert project.store.history(image["id"])[0]["reason"] == "find and replace"
