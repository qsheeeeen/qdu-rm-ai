template <typename Target, typename Param>
void Detector <Target, Param>::LoadParams(const std::string &path) {
	if (!PrepareParams(path)) {
		InitDefaultParams(path);
		PrepareParams(path);
		SPDLOG_WARN("Can not find parasm file. Created and reloaded.");
	}
	SPDLOG_DEBUG("Params loaded.");
}