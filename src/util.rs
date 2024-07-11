pub trait IntoRust {
    type RsType;
    fn rs(self) -> Self::RsType;
}
