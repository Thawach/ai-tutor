class CitationPolicy:
    """
    กำหนดว่าเมื่อไรควรแสดงแหล่งอ้างอิงแก่ผู้เรียน
    """

    def should_show(
        self,
        scaffolding_level: int,
        intervention: str | None = None,
    ) -> bool:

        if intervention == "misconception_correction":
            return True

        if scaffolding_level >= 3:
            return True

        return False